"""Piper TTS engine (onnxruntime, on-prem).

Piper is a fast, lightweight, high-quality neural TTS that runs on CPU with
onnxruntime - a good fit for a bank's on-premise infrastructure. Voice models
are downloaded once via ``python scripts/download_models.py`` into
``data/models/voices/`` (or pointed at directly via ``tts.model_path``).

The engine supports all piper-tts API generations (``wav_file`` kwarg, the
``speed`` kwarg, and the 1.6+ ``synthesize_wav`` + ``SynthesisConfig`` API).
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

from speechai.audio.io import AudioBuffer
from speechai.core.config import TTSConfig
from speechai.core.errors import ModelNotFoundError, SynthesisError
from speechai.core.metrics import model_load_seconds, model_loaded
from speechai.core.timing import Stopwatch, compute_rtf
from speechai.tts.base import SynthesisResult, TTSOptions

logger = logging.getLogger(__name__)


class PiperTTSEngine:
    """Piper TTS backed by the ``piper-tts`` package."""

    name = "piper"

    def __init__(self, config: TTSConfig, voices_dir: Path | None = None) -> None:
        self.config = config
        self._voices_dir = Path(voices_dir) if voices_dir else Path("data/models/voices")
        self._voice = None
        self._model_path: str | None = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._voice is not None:
            return
        try:
            from piper import PiperVoice  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ModelNotFoundError(
                "piper-tts is not installed. Run: pip install -e '.[engines]'"
            ) from exc
        model_path = self._resolve_model_path()
        config_path = model_path.with_suffix(model_path.suffix + ".json")
        stopwatch = Stopwatch()
        try:
            try:
                # piper-tts >= 1.6 accepts an explicit config path.
                self._voice = PiperVoice.load(
                    str(model_path), config_path=str(config_path) if config_path.is_file() else None
                )
            except TypeError:
                self._voice = PiperVoice.load(str(model_path))
        except Exception as exc:
            raise ModelNotFoundError(f"Could not load Piper voice {model_path}: {exc}") from exc
        model_load_seconds.observe(stopwatch.elapsed())
        model_loaded.labels(self.name).set(1)
        self._model_path = str(model_path)
        logger.info(
            "loaded Piper voice",
            extra={"voice": model_path.name, "load_seconds": round(stopwatch.elapsed(), 3)},
        )

    def close(self) -> None:
        self._voice = None
        model_loaded.labels(self.name).set(0)

    # ------------------------------------------------------------------
    def synthesize(self, text: str, options: TTSOptions | None = None) -> SynthesisResult:
        self.load()
        opts = options or TTSOptions()
        speed = opts.speed or self.config.default_speed
        stopwatch = Stopwatch()
        try:
            audio = self._synthesize_wav(self._voice, text, speed)
        except Exception as exc:
            raise SynthesisError(f"Piper synthesis failed: {exc}") from exc
        if audio.sample_rate != self.config.sample_rate:
            audio = audio.resample(self.config.sample_rate)
        latency = stopwatch.elapsed()
        return SynthesisResult(
            audio=audio,
            latency_seconds=latency,
            rtf=compute_rtf(latency, audio.duration_seconds),
            chars=len(text),
            engine=self.name,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _synthesize_wav(voice: object, text: str, speed: float) -> AudioBuffer:
        """Synthesize text to WAV bytes across piper-tts API generations."""
        length_scale = 1.0 / max(speed, 0.1)
        buffer = io.BytesIO()
        # 1) piper-tts < 1.3: synthesize(text, wav_file=..., speed=...)
        try:
            voice.synthesize(text, wav_file=buffer, speed=speed)  # type: ignore[attr-defined]
            buffer.seek(0)
            return AudioBuffer.from_wav_bytes(buffer.read())
        except TypeError:
            pass
        # 2) piper-tts ~1.3-1.5: synthesize(text, wav_file=..., length_scale=...)
        try:
            voice.synthesize(text, wav_file=buffer, length_scale=length_scale)  # type: ignore[attr-defined]
            buffer.seek(0)
            return AudioBuffer.from_wav_bytes(buffer.read())
        except TypeError:
            pass
        # 3) piper-tts >= 1.6: synthesize_wav(text, wave.Wave_write, SynthesisConfig)
        from piper import SynthesisConfig  # type: ignore[import-not-found]

        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(  # type: ignore[attr-defined]
                text,
                wav_file,
                syn_config=SynthesisConfig(length_scale=length_scale),
            )
        buffer.seek(0)
        return AudioBuffer.from_wav_bytes(buffer.read())

    # ------------------------------------------------------------------
    def _resolve_model_path(self) -> Path:
        if self.config.model_path:
            path = Path(self.config.model_path)
        else:
            path = self._voices_dir / f"{self.config.voice}.onnx"
        if not path.exists():
            raise ModelNotFoundError(
                f"Piper voice model not found at {path}. "
                f"Run `python scripts/download_models.py --piper-voice {self.config.voice}`."
            )
        return path
