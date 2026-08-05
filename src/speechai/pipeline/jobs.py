"""Batch job model and state machine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobType(str, Enum):
    transcribe = "transcribe"
    synthesize = "synthesize"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"

    TERMINAL = frozenset({"succeeded", "failed", "canceled"})


@dataclass
class Job:
    """A single batch job in the pipeline."""

    id: str
    type: JobType
    status: JobStatus = JobStatus.queued
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    input: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0

    def mark_started(self) -> None:
        self.status = JobStatus.running
        self.started_at = time.time()
        self.attempts += 1

    def mark_succeeded(self, result: dict[str, Any]) -> None:
        self.status = JobStatus.succeeded
        self.result = result
        self.finished_at = time.time()

    def mark_failed(self, error: str, *, details: dict[str, Any] | None = None) -> None:
        self.status = JobStatus.failed
        self.error = error
        self.finished_at = time.time()
        if details:
            self.metrics.update(details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input": self.input,
            "result": self.result,
            "error": self.error,
            "metrics": self.metrics,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            id=data["id"],
            type=JobType(data["type"]),
            status=JobStatus(data["status"]),
            created_at=data.get("created_at", 0.0),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            input=data.get("input") or {},
            result=data.get("result"),
            error=data.get("error"),
            metrics=data.get("metrics") or {},
            attempts=data.get("attempts", 0),
        )


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]
