"""STT engine interface, data types and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from speechai.audio.io import AudioBuffer
from speechai.core.config import Settings
from speechai.core.errors import ValidationError


@dataclass
class STTOptions:
    """Per-request options for STT engines."""

    language: str | None = None
    beam_size: int = 5
    # None = follow the engine config; True/False override per request.
    vad_filter: bool | None = None
    task: str = "transcribe"  # "transcribe" | "translate"


@dataclass
class Segment:
    """A single transcript segment with timing and confidence."""

    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[Segment]
    duration_seconds: float
    latency_seconds: float
    rtf: float
    engine: str
    avg_confidence: float | None = None
    no_speech_prob: float | None = None

    def to_dict(self, *, include_segments: bool = True) -> dict:
        data: dict = {
            "text": self.text,
            "language": self.language,
            "engine": self.engine,
            "avg_confidence": self.avg_confidence,
        }
        if include_segments:
            data["segments"] = [
                {"text": s.text, "start": s.start, "end": s.end, "confidence": s.confidence}
                for s in self.segments
            ]
        return data


class STTEngine(Protocol):
    """Speech-to-text engine contract.

    Implementations must be thread-safe enough for ``asyncio.to_thread`` use
    (faster-whisper models are safe for sequential calls per model instance).
    """

    name: str

    def load(self) -> None: ...
    def transcribe(self, audio: AudioBuffer, options: STTOptions | None = None) -> TranscriptionResult: ...
    def close(self) -> None: ...


def build_stt_engine(settings: Settings) -> STTEngine:
    """Factory for STT engines based on configuration."""
    engine_name = settings.stt.engine
    if engine_name == "whisper":
        from speechai.stt.whisper_engine import WhisperSTTEngine

        return WhisperSTTEngine(settings.stt)
    raise ValidationError(f"Unknown STT engine: {engine_name!r}")


def avg_segment_confidence(segments: list[Segment]) -> float | None:
    values = [s.confidence for s in segments if s.confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)
