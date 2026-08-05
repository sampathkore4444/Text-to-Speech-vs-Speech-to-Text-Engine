"""TTS engine interface, data types and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from speechai.audio.io import AudioBuffer
from speechai.core.config import Settings
from speechai.core.errors import ValidationError


@dataclass
class TTSOptions:
    """Per-request options for TTS engines."""

    speed: float = 1.0
    voice: str | None = None


@dataclass
class SynthesisResult:
    audio: AudioBuffer
    latency_seconds: float
    rtf: float
    chars: int
    engine: str


class TTSEngine(Protocol):
    """Text-to-speech engine contract (same threading notes as STT)."""

    name: str

    def load(self) -> None: ...
    def synthesize(self, text: str, options: TTSOptions | None = None) -> SynthesisResult: ...
    def close(self) -> None: ...


def build_tts_engine(settings: Settings) -> TTSEngine:
    """Factory for TTS engines based on configuration."""
    engine_name = settings.tts.engine
    if engine_name == "piper":
        from speechai.tts.piper_engine import PiperTTSEngine

        return PiperTTSEngine(settings.tts, voices_dir=settings.voices_dir)
    raise ValidationError(f"Unknown TTS engine: {engine_name!r}")
