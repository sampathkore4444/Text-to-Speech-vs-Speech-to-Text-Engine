"""Structured logging (JSON or plain text) with request/job correlation ids."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_job_id: ContextVar[str] = ContextVar("job_id", default="-")

_RESERVED_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message",
}


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


def set_job_id(value: str) -> None:
    _job_id.set(value)


def get_job_id() -> str:
    return _job_id.get()


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line with correlation context attached."""

    def __init__(self, service: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "service": self._service,
            "request_id": get_request_id(),
            "job_id": get_job_id(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "__dict__", {}).items():
            if key not in payload and key not in _RESERVED_ATTRS:
                payload[key] = value
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return json.dumps(
                {"ts": time.time(), "level": record.levelname, "message": record.getMessage()},
                default=str,
            )


def setup_logging(level: str = "INFO", fmt: str = "json", service: str = "bank-speech-ai") -> None:
    """Configure the root logger. Idempotent (replaces existing handlers)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JSONFormatter(service=service))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
