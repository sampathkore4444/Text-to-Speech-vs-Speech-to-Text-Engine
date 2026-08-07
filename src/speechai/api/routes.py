"""REST routes: transcription, synthesis, batch jobs, models."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from speechai.api.schemas import (
    JobMetrics,
    JobStatusResponse,
    JobSubmitResponse,
    ModelInfo,
    SynthesizeRequest,
    TranscribeResponse,
)
from speechai.core.errors import JobNotFoundError, PayloadTooLargeError, ValidationError
from speechai.core.logging import get_request_id
from speechai.pipeline.jobs import Job
from speechai.tts.streaming import StreamingSynthesizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["speech"])


# ---------------------------------------------------------------------------
# Synchronous low-latency endpoints
# ---------------------------------------------------------------------------
@router.post("/transcribe", response_model=TranscribeResponse, summary="Transcribe an audio file")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    redact: bool = Form(True),
    vad_filter: bool | None = Form(None),
) -> TranscribeResponse:
    settings = request.app.state.settings
    pipeline = request.app.state.pipeline
    content = await file.read()
    _validate_upload(content, settings.api.max_upload_mb)
    path = settings.uploads_dir / f"{uuid.uuid4().hex}.wav"
    path.write_bytes(content)
    logger.info(
        "transcribe request",
        extra={"size_bytes": len(content), "language": language, "vad_filter": vad_filter},
    )
    payload = await asyncio.to_thread(
        pipeline.transcribe_sync, path, language=language, redact=redact, vad_filter=vad_filter
    )
    return TranscribeResponse(**payload, request_id=get_request_id())


@router.post("/synthesize", summary="Synthesize speech from text (returns WAV)")
async def synthesize(
    request: Request,
    body: SynthesizeRequest,
    stream: bool = Query(False, description="Stream sentence-by-sentence for lower latency"),
) -> Response:
    pipeline = request.app.state.pipeline
    if stream:
        synthesizer = StreamingSynthesizer(pipeline.tts_engine, speed=body.speed)

        async def chunk_iterator() -> Any:
            async for chunk in synthesizer.synthesize(body.text):
                yield chunk

        return StreamingResponse(
            chunk_iterator(),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="speech_{int(time.time())}.wav"'},
        )
    output = await asyncio.to_thread(
        pipeline.synthesize_sync, body.text, voice=body.voice, speed=body.speed
    )
    return Response(
        content=output.audio.to_wav_bytes(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="speech_{int(time.time())}.wav"'},
    )


# ---------------------------------------------------------------------------
# Async batch job endpoints
# ---------------------------------------------------------------------------
@router.post("/jobs/transcribe", response_model=JobSubmitResponse, status_code=202, tags=["jobs"])
async def submit_transcribe_job(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    redact: bool = Form(True),
    vad_filter: bool | None = Form(None),
) -> JobSubmitResponse:
    settings = request.app.state.settings
    pipeline = request.app.state.pipeline
    content = await file.read()
    _validate_upload(content, settings.api.max_upload_mb)
    path = settings.uploads_dir / f"{uuid.uuid4().hex}.wav"
    path.write_bytes(content)
    job = await pipeline.submit_transcribe(path, language=language, redact=redact, vad_filter=vad_filter)
    logger.info("transcribe job submitted", extra={"job_id": job.id})
    return JobSubmitResponse(job_id=job.id, status=job.status.value, url=f"/v1/jobs/{job.id}")


@router.post("/jobs/synthesize", response_model=JobSubmitResponse, status_code=202, tags=["jobs"])
async def submit_synthesize_job(
    request: Request, body: SynthesizeRequest
) -> JobSubmitResponse:
    pipeline = request.app.state.pipeline
    job = await pipeline.submit_synthesize(body.text, voice=body.voice, speed=body.speed)
    logger.info("synthesize job submitted", extra={"job_id": job.id})
    return JobSubmitResponse(job_id=job.id, status=job.status.value, url=f"/v1/jobs/{job.id}")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job(request: Request, job_id: str) -> JobStatusResponse:
    pipeline = request.app.state.pipeline
    job = await pipeline.get_job(job_id)
    return _job_to_response(job)


@router.delete("/jobs/{job_id}", status_code=204, tags=["jobs"])
async def delete_job(request: Request, job_id: str) -> None:
    pipeline = request.app.state.pipeline
    await pipeline.delete_job(job_id)


@router.get("/jobs/{job_id}/audio", tags=["jobs"])
async def get_job_audio(request: Request, job_id: str) -> FileResponse:
    pipeline = request.app.state.pipeline
    job = await pipeline.get_job(job_id)
    artifact = pipeline.audio_artifact(job)
    if artifact is None:
        raise JobNotFoundError(f"Job {job_id} has no audio artifact")
    return FileResponse(artifact, media_type="audio/wav", filename=artifact.name)


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
@router.get("/models", response_model=dict[str, ModelInfo], tags=["system"])
async def models_info(request: Request) -> dict[str, ModelInfo]:
    pipeline = request.app.state.pipeline
    settings = request.app.state.settings
    loaded = pipeline.engine_status()
    return {
        "stt": ModelInfo(engine=settings.stt.engine, loaded=loaded["stt"], config=settings.stt.model_dump()),
        "tts": ModelInfo(engine=settings.tts.engine, loaded=loaded["tts"], config=settings.tts.model_dump()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_upload(content: bytes, max_mb: int) -> None:
    if not content:
        raise ValidationError("Uploaded file is empty")
    limit = max_mb * 1024 * 1024
    if len(content) > limit:
        raise PayloadTooLargeError(f"Upload exceeds the {max_mb} MB limit")


def _job_to_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        id=job.id,
        type=job.type.value,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        metrics=JobMetrics(**job.metrics) if job.metrics else None,
        result=job.result,
        audio_url=job.result.get("audio_url") if job.result else None,
    )
