"""Job queue backends.

- :class:`MemoryJobQueue` - in-process queue for single-process deployments
  and tests. Loop-agnostic (no asyncio primitives), so the API and a worker
  running in the same process/thread pool can share it.
- :class:`RedisJobQueue` - Redis-backed queue for horizontal scale-out (the
  production default in docker-compose).

Both implement the same :class:`JobQueue` protocol so the worker and API are
backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Protocol

from speechai.core.config import Settings
from speechai.pipeline.jobs import Job

logger = logging.getLogger(__name__)


class JobQueue(Protocol):
    async def enqueue(self, job: Job) -> None: ...
    async def dequeue(self, timeout: float | None = None) -> Job | None: ...
    async def get(self, job_id: str) -> Job | None: ...
    async def update(self, job: Job) -> None: ...
    async def delete(self, job_id: str) -> bool: ...
    async def depth(self) -> int: ...
    async def close(self) -> None: ...


class MemoryJobQueue:
    """In-memory queue with TTL pruning. Single-process only, loop-agnostic."""

    def __init__(self, *, ttl_seconds: int = 86400, max_results: int = 1000) -> None:
        self._jobs: dict[str, Job] = {}
        self._pending: deque[str] = deque()
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_results = max_results

    async def enqueue(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()
            self._pending.append(job.id)

    async def dequeue(self, timeout: float | None = None) -> Job | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                while self._pending:
                    job_id = self._pending.popleft()
                    job = self._jobs.get(job_id)
                    if job is not None:
                        return job
            if deadline is None:
                await asyncio.sleep(0.05)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(0.05, remaining))

    async def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    async def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    async def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    async def close(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _prune_locked(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        for job_id in [jid for jid, j in self._jobs.items() if j.created_at < cutoff]:
            del self._jobs[job_id]
        if len(self._jobs) > self._max_results:
            oldest = sorted(self._jobs, key=lambda jid: self._jobs[jid].created_at)
            for job_id in oldest[: len(oldest) - self._max_results]:
                del self._jobs[job_id]


class RedisJobQueue:
    """Redis-backed queue (BLPOP) for distributed deployments."""

    _QUEUE_KEY = "speech:queue"
    _JOB_KEY = "speech:job:{id}"

    def __init__(self, url: str, ttl_seconds: int = 86400) -> None:
        try:
            import redis.asyncio as aioredis  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("redis package is required for the Redis queue") from exc
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._ttl_seconds = ttl_seconds

    async def enqueue(self, job: Job) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._JOB_KEY.format(id=job.id), json.dumps(job.to_dict()), ex=self._ttl_seconds)
        pipe.rpush(self._QUEUE_KEY, job.id)
        await pipe.execute()

    async def dequeue(self, timeout: float | None = None) -> Job | None:
        result = await self._redis.blpop(self._QUEUE_KEY, timeout=timeout or 0)
        if result is None:
            return None
        _, job_id = result
        raw = await self._redis.get(self._JOB_KEY.format(id=job_id))
        if raw is None:
            return None
        return Job.from_dict(json.loads(raw))

    async def get(self, job_id: str) -> Job | None:
        raw = await self._redis.get(self._JOB_KEY.format(id=job_id))
        return Job.from_dict(json.loads(raw)) if raw else None

    async def update(self, job: Job) -> None:
        await self._redis.set(
            self._JOB_KEY.format(id=job.id), json.dumps(job.to_dict()), ex=self._ttl_seconds
        )

    async def delete(self, job_id: str) -> bool:
        return bool(await self._redis.delete(self._JOB_KEY.format(id=job_id)))

    async def depth(self) -> int:
        return int(await self._redis.llen(self._QUEUE_KEY) or 0)

    async def close(self) -> None:
        await self._redis.aclose()


def build_queue(settings: Settings) -> JobQueue:
    """Queue factory based on configuration."""
    if settings.queue.backend == "redis":
        logger.info("using Redis job queue", extra={"url": settings.queue.redis_url})
        return RedisJobQueue(settings.queue.redis_url, settings.storage.result_ttl_seconds)
    logger.info("using in-memory job queue")
    return MemoryJobQueue(
        ttl_seconds=settings.storage.result_ttl_seconds,
        max_results=settings.storage.max_results,
    )
