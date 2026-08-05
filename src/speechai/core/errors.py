"""Domain errors with stable machine-readable codes and HTTP status mapping."""

from __future__ import annotations

from typing import Any


class SpeechAIError(Exception):
    """Base class for all platform errors."""

    status_code = 500
    error_code = "internal_error"
    retryable = False

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.error_code, "message": self.message}
        if self.details is not None:
            body["details"] = self.details
        return body


class ValidationError(SpeechAIError):
    status_code = 400
    error_code = "validation_error"


class AudioFormatError(SpeechAIError):
    status_code = 400
    error_code = "audio_format_error"


class PayloadTooLargeError(SpeechAIError):
    status_code = 413
    error_code = "payload_too_large"


class JobNotFoundError(SpeechAIError):
    status_code = 404
    error_code = "job_not_found"


class UnauthorizedError(SpeechAIError):
    status_code = 401
    error_code = "unauthorized"


class QuotaExceededError(SpeechAIError):
    status_code = 429
    error_code = "quota_exceeded"


class EngineUnavailableError(SpeechAIError):
    status_code = 503
    error_code = "engine_unavailable"
    retryable = True


class ModelNotFoundError(EngineUnavailableError):
    error_code = "model_not_found"


class TranscriptionError(SpeechAIError):
    status_code = 500
    error_code = "transcription_failed"
    retryable = True


class SynthesisError(SpeechAIError):
    status_code = 500
    error_code = "synthesis_failed"
    retryable = True
