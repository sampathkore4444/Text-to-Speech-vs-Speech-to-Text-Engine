"""WebSocket routes: real-time streaming transcription and synthesis.

Transcription protocol: the client sends a JSON config message first, then
raw little-endian 16-bit PCM chunks (mono, 16 kHz recommended). The server
responds with JSON events: ``{"type": "partial", ...}`` live hypotheses and
``{"type": "final", ...}`` per completed utterance. Send ``{"action": "stop"}``
(or disconnect) to close the stream; the final utterance is flushed on close.

Synthesis protocol: the client sends ``{"text": "..."}``; the server streams
back one WAV blob per sentence.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from speechai.core import metrics
from speechai.core.errors import UnauthorizedError
from speechai.stt.streaming import StreamingTranscriber
from speechai.tts.streaming import StreamingSynthesizer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/v1/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    pipeline = websocket.app.state.pipeline
    await websocket.accept()
    metrics.ws_active.labels("transcribe").inc()
    transcriber: StreamingTranscriber | None = None
    try:
        first = await websocket.receive_json()
        if settings.api.api_key and first.get("api_key") != settings.api.api_key:
            raise UnauthorizedError("Invalid or missing API key")
        language = first.get("language") or settings.stt.language
        input_rate = int(first.get("sample_rate", 16000))
        partial_interval_ms = int(
            first.get("partial_interval_ms") or settings.stt.partial_interval_ms
        )
        # Engine loading blocks; do it off the event loop.
        engine = await asyncio.to_thread(lambda: pipeline.stt_engine)
        transcriber = StreamingTranscriber(
            engine,
            language=language,
            input_rate=input_rate,
            partial_interval_ms=partial_interval_ms,
            max_utterance_ms=settings.stt.max_segment_ms,
            min_silence_ms=settings.stt.min_silence_ms,
            vad_backend=settings.vad.backend,
        )
        while True:
            message = await websocket.receive()
            mtype = message.get("type")
            if mtype == "websocket.disconnect":
                break
            if message.get("text") is not None:
                data = json.loads(message["text"])
                if data.get("action") in ("stop", "close"):
                    break
            if message.get("bytes"):
                async for event in transcriber.feed(message["bytes"]):
                    await websocket.send_json(event.to_dict())
        async for event in transcriber.finish():
            await websocket.send_json(event.to_dict())
        await websocket.close()
    except WebSocketDisconnect:
        if transcriber is not None:
            try:
                async for event in transcriber.finish():  # best effort flush
                    await websocket.send_json(event.to_dict())
            except Exception:  # socket already gone; nothing to do
                pass
    except UnauthorizedError:
        await websocket.close(code=1008)
    except Exception:
        logger.exception("streaming transcription error")
        await websocket.close(code=1011)
    finally:
        metrics.ws_active.labels("transcribe").dec()


@router.websocket("/v1/ws/synthesize")
async def ws_synthesize(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    pipeline = websocket.app.state.pipeline
    await websocket.accept()
    metrics.ws_active.labels("synthesize").inc()
    try:
        config = await websocket.receive_json()
        if settings.api.api_key and config.get("api_key") != settings.api.api_key:
            raise UnauthorizedError("Invalid or missing API key")
        text = config.get("text", "")
        if not text.strip():
            raise ValueError("text must not be empty")
        engine = await asyncio.to_thread(lambda: pipeline.tts_engine)
        synthesizer = StreamingSynthesizer(engine, speed=float(config.get("speed", 1.0)))
        chunks = 0
        async for chunk in synthesizer.synthesize(text):
            await websocket.send_bytes(chunk)
            chunks += 1
        await websocket.send_text(json.dumps({"type": "done", "chunks": chunks}))
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except UnauthorizedError:
        await websocket.close(code=1008)
    except Exception:
        logger.exception("streaming synthesis error")
        await websocket.close(code=1011)
    finally:
        metrics.ws_active.labels("synthesize").dec()
