"""Batch worker process: consumes the job queue and executes batch jobs.

Run with: ``python -m speechai.workers.batch_worker``
Horizontal scale-out: start N worker replicas sharing the same Redis queue.
"""

from __future__ import annotations

import asyncio
import logging

from speechai.core import metrics
from speechai.core.config import Settings
from speechai.core.logging import set_job_id, setup_logging
from speechai.pipeline.batch import BatchPipeline
from speechai.pipeline.queue import build_queue

logger = logging.getLogger(__name__)


async def run(settings: Settings) -> None:
    queue = build_queue(settings)
    pipeline = BatchPipeline(settings, queue)
    logger.info("worker started", extra={"queue_backend": settings.queue.backend})
    try:
        while True:
            job = await queue.dequeue(timeout=settings.queue.poll_interval_seconds)
            metrics.speech_queue_depth.set(await queue.depth())
            if job is None:
                continue
            set_job_id(job.id)
            try:
                logger.info(
                    "processing job", extra={"job_id": job.id, "job_type": job.type.value}
                )
                job = await pipeline.run_job(job)
                logger.info(
                    "job finished",
                    extra={"job_id": job.id, "status": job.status.value},
                )
            finally:
                set_job_id("-")
    finally:
        await queue.close()
        logger.info("worker stopped")


def main() -> None:
    settings = Settings.load()
    setup_logging(settings.service.log_level, settings.service.log_format, settings.service.name)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("worker stopped by user")


if __name__ == "__main__":
    main()
