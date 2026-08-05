"""Batch pipeline tests: submission, execution, redaction, errors."""

from __future__ import annotations

import pytest

from speechai.core.errors import JobNotFoundError, ValidationError
from speechai.pipeline.jobs import JobStatus


async def test_synthesize_job_succeeds(pipeline, fake_tts) -> None:
    job = await pipeline.submit_synthesize("Hello bank customer")
    assert job.status == JobStatus.queued

    job = await pipeline.run_job(job)
    assert job.status == JobStatus.succeeded
    assert job.result["duration_seconds"] > 0
    assert job.result["audio_url"] == f"/v1/jobs/{job.id}/audio"
    assert fake_tts.calls == 1

    artifact = pipeline.audio_artifact(job)
    assert artifact is not None
    assert artifact.is_file()


async def test_transcribe_job_succeeds(pipeline, fake_stt, sample_audio) -> None:
    job = await pipeline.submit_transcribe(sample_audio)
    job = await pipeline.run_job(job)
    assert job.status == JobStatus.succeeded
    assert fake_stt.text in job.result["text"]
    assert job.metrics or job.result["metrics"]


async def test_transcribe_job_redacts(pipeline, fake_stt, sample_audio) -> None:
    fake_stt.text = "My card number is 4242 4242 4242 4242"
    job = await pipeline.submit_transcribe(sample_audio)
    job = await pipeline.run_job(job)
    assert job.status == JobStatus.succeeded
    assert job.result["redacted"] is True
    assert "4242 4242 4242 4242" not in job.result["text"]


async def test_get_job_not_found(pipeline) -> None:
    with pytest.raises(JobNotFoundError):
        await pipeline.get_job("does-not-exist")


async def test_delete_job(pipeline, fake_stt, sample_audio) -> None:
    job = await pipeline.submit_transcribe(sample_audio)
    assert await pipeline.delete_job(job.id) is True
    with pytest.raises(JobNotFoundError):
        await pipeline.delete_job(job.id)


async def test_empty_synthesis_rejected(pipeline) -> None:
    with pytest.raises(ValidationError):
        await pipeline.submit_synthesize("   ")


async def test_queue_depth(pipeline, fake_stt, sample_audio) -> None:
    await pipeline.submit_transcribe(sample_audio)
    await pipeline.submit_transcribe(sample_audio)
    assert await pipeline.queue.depth() == 2
