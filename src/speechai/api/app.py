"""FastAPI application factory and entry point.

``create_app`` wires settings, logging, the queue, the batch pipeline, the
redactor, middlewares and exception handlers. Engines stay lazy (loaded on
first use) so the API boots instantly and health checks work without models.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from speechai import __version__
from speechai.api.middleware import APIKeyMiddleware, ObservabilityMiddleware
from speechai.api.routes import router as api_router
from speechai.api.schemas import HealthResponse
from speechai.api.ws import router as ws_router
from speechai.api.ws_openapi import extend_openapi_schema
from speechai.core import metrics
from speechai.core.config import Settings
from speechai.core.errors import SpeechAIError
from speechai.core.logging import setup_logging
from speechai.pipeline.batch import BatchPipeline
from speechai.pipeline.queue import build_queue
from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.stt.base import STTEngine
from speechai.tts.base import TTSEngine

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    stt_engine: STTEngine | None = None,
    tts_engine: TTSEngine | None = None,
) -> FastAPI:
    """Application factory. Engines can be injected for tests."""
    settings = settings or Settings.load()
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(
            settings.service.log_level, settings.service.log_format, settings.service.name
        )
        queue = build_queue(settings)
        redactor = Redactor(RedactionPolicy.from_settings(settings.redaction))
        app.state.settings = settings
        app.state.queue = queue
        app.state.redactor = redactor
        app.state.pipeline = BatchPipeline(
            settings,
            queue,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            redactor=redactor,
        )
        logger.info(
            "service started",
            extra={
                "version": __version__,
                "environment": settings.service.environment,
                "queue_backend": settings.queue.backend,
            },
        )
        yield
        await queue.close()
        logger.info("service stopped")

    app = FastAPI(
        title="Bank Speech AI",
        version=__version__,
        description="Production-grade STT + TTS platform for banking: ASR with VAD, "
        "PII redaction, streaming, batch jobs and evaluation.",
        lifespan=lifespan,
    )

    # Middleware (last added runs first).
    app.add_middleware(ObservabilityMiddleware)
    if settings.api.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=settings.api.api_key)

    app.include_router(api_router)
    app.include_router(ws_router)

    _register_exception_handlers(app)

    # FastAPI omits WebSocket routes from the generated OpenAPI schema; inject
    # structured path entries (x-websocket vendor extension) so they show up in
    # /docs and /openapi.json. See speechai.api.ws_openapi and docs/ws-protocol.md.
    _openapi = app.openapi

    def _openapi_with_ws() -> dict:
        if app.openapi_schema is None:
            app.openapi_schema = extend_openapi_schema(_openapi())
        return app.openapi_schema

    app.openapi = _openapi_with_ws

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        pipeline = request.app.state.pipeline
        queue = request.app.state.queue
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=settings.service.environment,
            models=pipeline.engine_status(),
            queue={"backend": settings.queue.backend, "depth": await queue.depth()},
        )

    @app.get("/metrics", include_in_schema=False, tags=["system"])
    async def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    # Browser demo console (transcribe upload / live mic / TTS player).
    _ui_file = Path(__file__).parent / "ui" / "index.html"

    @app.get("/", include_in_schema=False, tags=["ui"])
    async def demo_ui() -> FileResponse:
        return FileResponse(_ui_file, media_type="text/html")

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SpeechAIError)
    async def speechai_error_handler(request: Request, exc: SpeechAIError) -> JSONResponse:
        metrics.errors_total.labels("api", exc.error_code).inc()
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        metrics.errors_total.labels("api", "unhandled").inc()
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )


app = create_app()
