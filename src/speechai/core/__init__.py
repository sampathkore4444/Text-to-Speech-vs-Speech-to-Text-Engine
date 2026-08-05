"""Core cross-cutting infrastructure: config, logging, metrics, errors, timing."""

from speechai.core.config import Settings
from speechai.core.errors import (
    AudioFormatError,
    EngineUnavailableError,
    JobNotFoundError,
    ModelNotFoundError,
    PayloadTooLargeError,
    QuotaExceededError,
    SpeechAIError,
    SynthesisError,
    TranscriptionError,
    UnauthorizedError,
    ValidationError,
)
from speechai.core.logging import setup_logging
from speechai.core.timing import Stopwatch, compute_rtf

__all__ = [
    "AudioFormatError",
    "EngineUnavailableError",
    "JobNotFoundError",
    "ModelNotFoundError",
    "PayloadTooLargeError",
    "QuotaExceededError",
    "Settings",
    "SpeechAIError",
    "Stopwatch",
    "SynthesisError",
    "TranscriptionError",
    "UnauthorizedError",
    "ValidationError",
    "compute_rtf",
    "setup_logging",
]
