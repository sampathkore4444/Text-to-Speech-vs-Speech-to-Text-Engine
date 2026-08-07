"""Batch transcription/synthesis pipeline.

Owns the job lifecycle (submit -> run -> result), computes quality/latency
metrics (RTF, latency), emits Prometheus metrics and keeps audio artifacts on
disk for retrieval and audit.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speechai.audio.io import AudioBuffer, load_audio, to_asr_audio, write_wav
from speechai.core import metrics
from speechai.core.config import Settings
from speechai.core.errors import JobNotFoundError, ValidationError
from speechai.core.timing import Stopwatch
from speechai.pipeline.jobs import Job, JobType, new_job_id
from speechai.pipeline.queue import JobQueue
from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.stt.base import STTEngine, STTOptions, build_stt_engine
from speechai.stt.postprocess import TextPostProcessor, refine_segments
from speechai.tts.base import TTSEngine, TTSOptions, build_tts_engine
from speechai.tts.textnorm import TextNormalizer

logger = logging.getLogger(__name__)


@dataclass
class SynthOutput:
    """Output of a synchronous TTS call, including the audio for persistence."""

    audio: AudioBuffer
    latency_seconds: float
    rtf: float
    chars: int
    redacted: bool = False

    @property
    def audio_duration_seconds(self) -> float:
        return self.audio.duration_seconds


class BatchPipeline:
    """Submit + execute batch jobs; also exposes synchronous helpers used by
    the low-latency REST endpoints."""

    def __init__(
        self,
        settings: Settings,
        queue: JobQueue,
        *,
        stt_engine: STTEngine | None = None,
        tts_engine: TTSEngine | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.settings = settings
        self.queue = queue
        self._injected_stt = stt_engine
        self._injected_tts = tts_engine
        self._stt: STTEngine | None = None
        self._tts: TTSEngine | None = None
        self._engine_lock = threading.Lock()
        self.redactor = redactor or Redactor(RedactionPolicy.from_settings(settings.redaction))
        self.postprocessor = TextPostProcessor(self.redactor)
        self.text_normalizer = TextNormalizer(self.redactor)

    # ------------------------------------------------------------------
    # Engine access (lazy singletons)
    # ------------------------------------------------------------------
    @property
    def stt_engine(self) -> STTEngine:
        if self._stt is None:
            with self._engine_lock:
                if self._stt is None:
                    self._stt = self._injected_stt or build_stt_engine(self.settings)
        return self._stt

    @property
    def tts_engine(self) -> TTSEngine:
        if self._tts is None:
            with self._engine_lock:
                if self._tts is None:
                    self._tts = self._injected_tts or build_tts_engine(self.settings)
        return self._tts

    def engine_status(self) -> dict[str, bool]:
        return {
            "stt": self._stt is not None or self._injected_stt is not None,
            "tts": self._tts is not None or self._injected_tts is not None,
        }

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    async def submit_transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        redact: bool = True,
    ) -> Job:
        job = Job(
            id=new_job_id(),
            type=JobType.transcribe,
            input={"audio_path": str(audio_path), "language": language, "redact": redact},
        )
        await self.queue.enqueue(job)
        return job

    async def submit_synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Job:
        if not text or not text.strip():
            raise ValidationError("Synthesis text must not be empty")
        job = Job(
            id=new_job_id(),
            type=JobType.synthesize,
            input={"text": text, "voice": voice, "speed": speed},
        )
        await self.queue.enqueue(job)
        return job

    async def get_job(self, job_id: str) -> Job:
        job = await self.queue.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job {job_id} not found or expired")
        return job

    async def delete_job(self, job_id: str) -> bool:
        deleted = await self.queue.delete(job_id)
        if not deleted:
            raise JobNotFoundError(f"Job {job_id} not found or expired")
        return True

    # ------------------------------------------------------------------
    # Execution (called by the batch worker)
    # ------------------------------------------------------------------
    async def run_job(self, job: Job) -> Job:
        job.mark_started()
        await self.queue.update(job)
        metrics.speech_jobs_active.inc()
        try:
            if job.type == JobType.transcribe:
                result = await asyncio.to_thread(self._execute_transcribe, job)
            else:
                result = await asyncio.to_thread(self._execute_synthesize, job)
            job.mark_succeeded(result)
        except ValidationError as exc:
            job.mark_failed(exc.message)
            metrics.errors_total.labels("pipeline", exc.error_code).inc()
        except Exception as exc:
            logger.exception("job failed", extra={"job_id": job.id, "job_type": job.type.value})
            job.mark_failed(repr(exc))
            metrics.errors_total.labels("pipeline", "internal").inc()
        finally:
            await self.queue.update(job)
            metrics.speech_jobs_active.dec()
            metrics.speech_jobs_total.labels(job.type.value, job.status.value).inc()
        return job

    # ------------------------------------------------------------------
    # Synchronous helpers (used by low-latency REST endpoints)
    # ------------------------------------------------------------------
    def transcribe_sync(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        redact: bool = True,
    ) -> dict[str, Any]:
        stopwatch = Stopwatch()
        audio = load_audio(audio_path)
        asr_audio = to_asr_audio(audio)
        options = STTOptions(
            language=language or self.settings.stt.language,
            beam_size=self.settings.stt.beam_size,
            vad_filter=self.settings.stt.vad_filter,
        )
        result = self.stt_engine.transcribe(asr_audio, options)
        processed = self.postprocessor.process(result.text, redact=redact) if redact else None
        # Split engine segments into per-sentence rows so multi-sentence audio
        # renders as one line per sentence (faster-whisper often emits a single
        # segment for an entire file). See postprocess.refine_segments.
        segments = refine_segments(result.segments) if result.segments else []
        payload: dict[str, Any] = {
            "text": processed.text if processed else result.text,
            "language": result.language,
            "engine": result.engine,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end, "confidence": s.confidence}
                for s in segments
            ],
            "redacted": bool(processed and processed.redacted),
            "redactions": [
                {"type": f.pii_type, "masked": f.masked} for f in (processed.findings if processed else [])
            ],
            "metrics": {
                "latency_seconds": stopwatch.elapsed(),
                "engine_seconds": result.latency_seconds,
                "rtf": result.rtf,
                "audio_duration_seconds": result.duration_seconds,
                "confidence": result.avg_confidence,
            },
        }
        metrics.stt_requests_total.labels("success", "sync").inc()
        metrics.stt_audio_seconds_total.labels("sync").inc(result.duration_seconds)
        metrics.stt_latency_seconds.observe(result.latency_seconds)
        metrics.stt_rtf.observe(result.rtf)
        if result.avg_confidence is not None:
            metrics.stt_confidence.observe(result.avg_confidence)
        return payload

    def synthesize_sync(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> SynthOutput:
        if not text or not text.strip():
            raise ValidationError("Synthesis text must not be empty")
        normalized = self.text_normalizer.normalize(text)
        result = self.tts_engine.synthesize(
            normalized.text, TTSOptions(speed=speed, voice=voice)
        )
        output = SynthOutput(
            audio=result.audio,
            latency_seconds=result.latency_seconds,
            rtf=result.rtf,
            chars=result.chars,
            redacted=normalized.redacted,
        )
        metrics.tts_requests_total.labels("success", "sync").inc()
        metrics.tts_chars_total.labels("sync").inc(result.chars)
        metrics.tts_latency_seconds.observe(result.latency_seconds)
        metrics.tts_rtf.observe(result.rtf)
        return output

    # ------------------------------------------------------------------
    # Job execution internals
    # ------------------------------------------------------------------
    def _execute_transcribe(self, job: Job) -> dict[str, Any]:
        return self.transcribe_sync(
            Path(job.input["audio_path"]),
            language=job.input.get("language"),
            redact=bool(job.input.get("redact", True)),
        )

    def _execute_synthesize(self, job: Job) -> dict[str, Any]:
        output = self.synthesize_sync(
            job.input.get("text", ""),
            voice=job.input.get("voice"),
            speed=float(job.input.get("speed", 1.0)),
        )
        artifact = self.artifact_path(job)
        write_wav(artifact, output.audio)
        return {
            "audio_path": str(artifact),
            "audio_url": f"/v1/jobs/{job.id}/audio",
            "duration_seconds": output.audio_duration_seconds,
            "redacted": output.redacted,
            "metrics": {
                "latency_seconds": output.latency_seconds,
                "rtf": output.rtf,
                "chars": output.chars,
            },
        }

    def artifact_path(self, job: Job) -> Path:
        return self.settings.results_dir / f"{job.id}.wav"

    def audio_artifact(self, job: Job) -> Path | None:
        if job.type != JobType.synthesize or not job.result:
            return None
        path = Path(job.result.get("audio_path", ""))
        return path if path.is_file() else None
