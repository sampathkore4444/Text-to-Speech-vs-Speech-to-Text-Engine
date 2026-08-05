"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SegmentOut(BaseModel):
    text: str
    start: float
    end: float
    confidence: float | None = None


class TranscribeMetrics(BaseModel):
    latency_seconds: float
    engine_seconds: float
    rtf: float
    audio_duration_seconds: float
    confidence: float | None = None


class RedactionOut(BaseModel):
    type: str
    masked: str


class TranscribeResponse(BaseModel):
    text: str
    language: str | None
    engine: str
    segments: list[SegmentOut] = Field(default_factory=list)
    redacted: bool = False
    redactions: list[RedactionOut] = Field(default_factory=list)
    metrics: TranscribeMetrics
    request_id: str = "-"


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    url: str


class JobMetrics(BaseModel):
    latency_seconds: float | None = None
    rtf: float | None = None
    audio_duration_seconds: float | None = None
    confidence: float | None = None
    chars: int | None = None


class JobStatusResponse(BaseModel):
    id: str
    type: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    metrics: JobMetrics | None = None
    result: dict[str, Any] | None = None
    audio_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    models: dict[str, bool]
    queue: dict[str, Any]


class ModelInfo(BaseModel):
    engine: str
    loaded: bool
    config: dict[str, Any]
