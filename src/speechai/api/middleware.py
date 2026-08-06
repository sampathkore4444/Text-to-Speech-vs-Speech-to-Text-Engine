"""ASGI middlewares: observability (request id + metrics) and optional API-key auth."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from speechai.core import metrics
from speechai.core.logging import set_request_id

_WHITELIST = frozenset({"/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json"})


class ObservabilityMiddleware:
    """Attaches a request id, times the request and records HTTP metrics."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:12]
        set_request_id(request_id)
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            metrics.http_latency_seconds.observe(time.perf_counter() - start)
            path = _template_path(scope.get("path", ""))
            metrics.http_requests_total.labels(
                scope.get("method", ""), path, str(status["code"])
            ).inc()
            set_request_id("-")


class APIKeyMiddleware:
    """Optional ``X-API-Key`` auth for everything except whitelisted paths."""

    def __init__(self, app: Any, api_key: str, whitelist: frozenset[str] = _WHITELIST) -> None:
        self.app = app
        self.api_key = api_key
        self.whitelist = whitelist

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "").split("?")[0]
        if path in self.whitelist:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        provided = headers.get(b"x-api-key", b"").decode()
        if provided != self.api_key:
            metrics.errors_total.labels("api", "unauthorized").inc()
            body = json.dumps(
                {"error": {"code": "unauthorized", "message": "Invalid or missing API key"}}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def _template_path(path: str) -> str:
    """Lower cardinality metric labels: dynamic segments (numbers, uuid-like
    hex job ids) collapse to {id} so each job/upload does not create a new
    Prometheus time series."""
    def _is_id(seg: str) -> bool:
        return seg.isdigit() or (len(seg) == 16 and all(c in "0123456789abcdef" for c in seg.lower()))

    segments = ["{id}" if _is_id(seg) else seg for seg in path.split("/")]
    return "/".join(segments)
