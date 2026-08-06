"""OpenAPI documentation for the WebSocket endpoints.

FastAPI does not render WebSocket routes in its generated OpenAPI schema.
This module injects structured path entries for ``/v1/ws/transcribe`` and
``/v1/ws/synthesize`` (marked with the vendor extension ``x-websocket: true``)
so they appear in ``/docs`` and ``/openapi.json``, with a pointer to the full
wire-protocol spec in ``docs/ws-protocol.md``.

Vendor extensions are the only standards-compliant way to describe WebSockets
in OpenAPI 3.x; clients must read them explicitly.
"""

from __future__ import annotations

from typing import Any

WS_SPEC_URL = "docs/ws-protocol.md"

_WS_PATHS: dict[str, Any] = {
    "/v1/ws/transcribe": {
        "get": {
            "operationId": "wsTranscribe",
            "summary": "WebSocket — real-time streaming ASR",
            "description": (
                "Live speech-to-text. After the handshake the client MUST send one JSON "
                "config message first: {\"sample_rate\": 16000, \"language\": \"en\", "
                "\"api_key\": \"...\", \"partial_interval_ms\": 2500}. It then streams raw "
                "little-endian 16-bit PCM (mono) audio as binary frames. The server replies "
                "with JSON events: {\"type\": \"partial\", ...} live hypotheses and "
                "{\"type\": \"final\", ...} per VAD-completed utterance (post-processed and "
                "PII-redacted). End with {\"action\": \"stop\"} or disconnect to flush the "
                "final utterance and close. Close codes: 1008 = auth failure, 1011 = "
                "internal error. Full wire spec: docs/ws-protocol.md."
            ),
            "x-websocket": True,
            "x-wire-protocol": WS_SPEC_URL,
            # OpenAPI 3.x requires `responses` on every operation object.
            "responses": {},
        }
    },
    "/v1/ws/synthesize": {
        "get": {
            "operationId": "wsSynthesize",
            "summary": "WebSocket — real-time streaming TTS",
            "description": (
                "Live text-to-speech. Send one JSON message: {\"text\": \"...\", "
                "\"speed\": 1.0, \"api_key\": \"...\"}. The server replies with one binary "
                "WAV frame per sentence (clients can start playback immediately) and "
                "finishes with {\"type\": \"done\", \"chunks\": N} before closing. Close "
                "codes: 1008 = auth failure, 1011 = internal error. Full wire spec: "
                "docs/ws-protocol.md."
            ),
            "x-websocket": True,
            "x-wire-protocol": WS_SPEC_URL,
            "responses": {},
        }
    },
}


def extend_openapi_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Merge the WebSocket path definitions into an OpenAPI schema (in place)."""
    paths = schema.setdefault("paths", {})
    for path, definition in _WS_PATHS.items():
        paths[path] = definition
    schema["x-websocket-spec"] = WS_SPEC_URL
    return schema
