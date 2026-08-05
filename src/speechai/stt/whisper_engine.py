"""faster-whisper STT engine (CTranslate2, on-prem, CPU/GPU).

faster-whisper runs Whisper models via CTranslate2 with int8 quantization on
CPU, giving real-time factors well under 1.0 for small/base models - ideal
for a bank that must keep customer audio on-premise.
"""

from __future__ import annotations

import logging
import math

from speechai.audio.io import ASR_SAMPLE_RATE, AudioBuffer
from speechai.core.config import STTConfig
from speechai.core.errors import ModelNotFoundError, TranscriptionError
from speechai.core.metrics import model_load_seconds, model_loaded
from speechai.core.timing import Stopwatch, compute_rtf
from speechai.stt.base import Segment, STTOptions, TranscriptionResult, avg_segment_confidence

logger = logging.getLogger(__name__)


class WhisperSTTEngine:
    """Whisper STT backed by faster-whisper."""

    name = "faster-whisper"

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model = None
        self._device: str | None = None
        self._compute_type: str | None = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ModelNotFoundError(
                "faster-whisper is not installed. Run: pip install -e '.[engines]'"
            ) from exc
        model_ref = self._model_ref()
        device, compute_type = _resolve_device(self.config.device, self.config.compute_type)
        stopwatch = Stopwatch()
        try:
            # A local directory (fine-tuned CTranslate2 export) or an HF hub
            # id; otherwise the model name downloads from the Hub on first use.
            self._model = WhisperModel(model_ref, device=device, compute_type=compute_type)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Could not load Whisper model {model_ref!r}: {exc}"
            ) from exc
        model_load_seconds.observe(stopwatch.elapsed())
        model_loaded.labels(self.name).set(1)
        self._device = device
        self._compute_type = compute_type
        logger.info(
            "loaded Whisper model",
            extra={
                "model": model_ref,
                "device": device,
                "compute_type": compute_type,
                "load_seconds": round(stopwatch.elapsed(), 3),
            },
        )

    def _model_ref(self) -> str:
        """The model to load: a converted CTranslate2 directory, or a size name."""
        return self.config.model_path or self.config.model_size

    def close(self) -> None:
        self._model = None
        model_loaded.labels(self.name).set(0)

    # ------------------------------------------------------------------
    def transcribe(self, audio: AudioBuffer, options: STTOptions | None = None) -> TranscriptionResult:
        self.load()
        opts = options or STTOptions()
        # faster-whisper expects mono float32 @ 16 kHz.
        if audio.sample_rate != ASR_SAMPLE_RATE:
            audio = audio.resample(ASR_SAMPLE_RATE)
        stopwatch = Stopwatch()
        try:
            segments_iter, info = self._model.transcribe(
                audio.samples,
                language=opts.language or self.config.language,
                beam_size=opts.beam_size or self.config.beam_size,
                vad_filter=opts.vad_filter if opts.vad_filter is not None else self.config.vad_filter,
                task=opts.task,
            )
            segments = [
                Segment(
                    text=seg.text.strip(),
                    start=float(seg.start),
                    end=float(seg.end),
                    confidence=_logprob_to_confidence(seg.avg_logprob),
                )
                for seg in segments_iter
            ]
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc
        latency = stopwatch.elapsed()
        text = " ".join(seg.text for seg in segments).strip()
        language = getattr(info, "language", None)
        duration = float(getattr(info, "duration", audio.duration_seconds) or audio.duration_seconds)
        return TranscriptionResult(
            text=text,
            language=language,
            segments=segments,
            duration_seconds=duration,
            latency_seconds=latency,
            rtf=compute_rtf(latency, duration),
            engine=self.name,
            avg_confidence=avg_segment_confidence(segments),
        )


def _logprob_to_confidence(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    return round(float(math.exp(logprob)), 4)


def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    if device == "auto":
        try:
            import torch  # noqa: F401

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type
