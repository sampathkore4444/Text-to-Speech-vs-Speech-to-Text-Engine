# WebSocket Wire Protocol — Formal Specification

> The machine-level contract for the two real-time endpoints:
>
> | Endpoint | Purpose | Server code |
> |---|---|---|
> | `/v1/ws/transcribe` | Real-time streaming ASR (speech → text) | `speechai.api.ws.ws_transcribe` |
> | `/v1/ws/synthesize` | Real-time streaming TTS (text → speech) | `speechai.api.ws.ws_synthesize` |
>
> Companion documents: `END-TO-END-FLOW.md` (Section 6 — client/server
> communication, Section 5.0.4/5.0.5 — sequence diagrams) and the generated
> OpenAPI document at `GET /openapi.json` (the endpoints are injected into it
> with the vendor extension `x-websocket: true` — see Section 6 here).

---

## 1. Conventions

- **Transport:** standard RFC 6455 WebSockets. Both endpoints are **server
  closes** — the client never initiates the close except by disconnecting.
- **Framing:** two frame kinds are used:
  - **JSON text frames** — all control/config messages and all server events.
  - **Binary frames** — audio payloads only (PCM16 for ASR input, WAV for TTS
    output). Binary frames are never JSON.
- **Byte order:** PCM audio is **little-endian** (`int16`, mono).
- **Auth:** if `api.api_key` is configured, the client MUST include
  `"api_key": "<key>"` in its **first** JSON message. There are no auth headers
  in the WebSocket handshake. Auth failure → close code **1008**.
- **Close codes used by the server:**
  - `1000` — normal completion (after `done` / flush).
  - `1008` — unauthorized (missing/wrong `api_key`).
  - `1011` — internal error (engine failure, bad config, non-JSON first frame).
- **Correlation:** server logs for the lifetime of a connection are tagged with
  the standard `request_id`; job-level logs use `job_id`.

---

## 2. Endpoint: `/v1/ws/transcribe` (streaming ASR)

### 2.1 Lifecycle

```
 open ──► (1) client sends config JSON ──► (2) client streams PCM16 binary frames
                                              server emits partial/final JSON events
                                          (3) client sends {"action":"stop"} or disconnects
                                              server flushes last final ──► close(1000)
```

### 2.2 Client → server messages

**Message 1 — config (REQUIRED, JSON text frame, must be the first message).**

```json
{
  "sample_rate": 16000,
  "language": "en",
  "api_key": "...",
  "partial_interval_ms": 2500
}
```

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `sample_rate` | integer | no | `16000` | Sample rate of the PCM the client will send. Values other than 16000 are resampled server-side to 16 kHz |
| `language` | string \| null | no | server `stt.language` | Whisper language code, e.g. `en`. `null`/omitted = auto-detect |
| `api_key` | string | if auth enabled | — | Must equal `api.api_key` or the connection is closed with `1008` |
| `partial_interval_ms` | integer | no | server `stt.partial_interval_ms` (2500) | Cadence of `partial` events in milliseconds |

**Messages 2..N — audio (binary frames, repeatable).**

Raw **little-endian 16-bit PCM, mono** at `sample_rate`. Chunk size is
arbitrary; the server accumulates into fixed 30 ms VAD frames. Sending nothing
is legal but yields no transcription.

**Stop message — JSON text frame.**

```json
{ "action": "stop" }
```

`"action"` accepts `"stop"` or `"close"`. On receipt the server flushes any
in-progress utterance as a final `final` event, then closes with `1000`.
Disconnecting has the same effect (best-effort flush).

> Any other JSON text frame is ignored. A non-JSON text frame, or a binary
> frame sent before the config message, is an internal error → close `1011`.

### 2.3 Server → client events (JSON text frames)

**Partial event** (live hypothesis; emitted at most once per
`partial_interval_ms` while an utterance is in progress — first one at
`partial_interval_ms` after stream start). Text is whitespace-cleaned but
**not** PII-redacted (avoids flickering masks). `segments` is absent.

```json
{
  "type": "partial",
  "text": "your account ba",
  "start": 0.0,
  "end": 1.2,
  "utterance_index": 1,
  "confidence": 0.9
}
```

**Final event** (one per completed utterance, VAD-segmented; cleaned **and**
PII-redacted). `utterance_index` increments per final.

```json
{
  "type": "final",
  "text": "Your account balance is one thousand two hundred dollars.",
  "start": 0.0,
  "end": 4.2,
  "utterance_index": 1,
  "confidence": 0.94,
  "segments": [
    {"text": "Your account balance is one thousand two hundred dollars.", "start": 0.0, "end": 4.2, "confidence": 0.94}
  ]
}
```

**Event schema (JSON Schema draft 2020-12):**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://bank-speech-ai/schemas/transcribe-event.json",
  "title": "StreamingTranscribeEvent",
  "oneOf": [
    {
      "title": "Partial",
      "type": "object",
      "required": ["type", "text", "start", "end", "utterance_index"],
      "properties": {
        "type": {"const": "partial"},
        "text": {"type": "string"},
        "start": {"type": "number", "minimum": 0},
        "end": {"type": "number", "minimum": 0},
        "utterance_index": {"type": "integer", "minimum": 1},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1}
      }
    },
    {
      "title": "Final",
      "type": "object",
      "required": ["type", "text", "start", "end", "utterance_index", "segments"],
      "properties": {
        "type": {"const": "final"},
        "text": {"type": "string"},
        "start": {"type": "number", "minimum": 0},
        "end": {"type": "number", "minimum": 0},
        "utterance_index": {"type": "integer", "minimum": 1},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "segments": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["text", "start", "end"],
            "properties": {
              "text": {"type": "string"},
              "start": {"type": "number", "minimum": 0},
              "end": {"type": "number", "minimum": 0},
              "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1}
            }
          }
        }
      }
    }
  ]
}
```

### 2.4 Segmentation behavior (server-side, for client expectations)

- Speech must persist ≥ `vad.min_speech_ms` (250 ms) before an utterance opens
  (no click-triggered starts).
- Silence ≥ `stt.min_silence_ms` (500 ms) closes an utterance (trailing silence
  trimmed).
- A hard `stt.max_segment_ms` (12 s) cap bounds utterance length.
- Frame VAD: `vad.backend` = `webrtc` (preferred, `aggressiveness=2`, 30 ms
  frames) or `energy` (`-35 dB` fallback).

### 2.5 Worked example (Python, `websockets`)

```python
import asyncio, json, websockets

async def stream_asr(pcm16_chunks, url="ws://localhost:8000/v1/ws/transcribe", api_key=None):
    config = {"sample_rate": 16000, "language": "en"}
    if api_key:
        config["api_key"] = api_key
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(config))
        for chunk in pcm16_chunks:            # bytes: int16 LE mono @ 16 kHz
            await ws.send(chunk)
        await ws.send(json.dumps({"action": "stop"}))
        events = []
        try:
            async for message in ws:
                events.append(json.loads(message))   # partial/final JSON events
        except websockets.ConnectionClosedOK:
            pass
        return events

asyncio.run(stream_asr(iter([b"\x00\x00" * 16000])))
```

---

## 3. Endpoint: `/v1/ws/synthesize` (streaming TTS)

### 3.1 Lifecycle

```
 open ──► (1) client sends one JSON request
          (2) server sends one binary WAV frame per sentence
          (3) server sends {"type":"done","chunks":N} ──► close(1000)
```

### 3.2 Client → server message (single JSON text frame, first and only)

```json
{
  "text": "Your balance is one thousand dollars. Thank you.",
  "speed": 1.0,
  "api_key": "..."
}
```

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `text` | string | yes | — | Text to synthesize. Must be non-empty (min length 1) |
| `speed` | number | no | `1.0` | Speaking rate. **Supported range 0.5–2.0** — enforced by the REST endpoint; on this WebSocket endpoint the server does not clamp the value, so treat the range as a client-side contract |
| `api_key` | string | if auth enabled | — | As in Section 2.2 |

### 3.3 Server → client frames

**Binary frames (one per sentence):** a complete WAV file (`RIFF` header,
PCM16, mono) at `tts.sample_rate` (default **22050 Hz**). Sentences are split
server-side on `[.!?]` + whitespace. The client can begin playback as soon as
the first frame arrives — later frames may still be synthesizing.

**Terminal JSON frame:**

```json
{ "type": "done", "chunks": 3 }
```

`chunks` = number of WAV frames sent. The server closes with `1000` immediately
after.

### 3.4 Schemas

**Request (JSON Schema draft 2020-12):**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://bank-speech-ai/schemas/synthesize-request.json",
  "title": "StreamingSynthesizeRequest",
  "type": "object",
  "required": ["text"],
  "properties": {
    "text": {"type": "string", "minLength": 1, "maxLength": 10000},
    "speed": {"type": "number", "default": 1.0, "description": "Speaking rate; supported range 0.5-2.0 (client-side contract - not clamped by the server on this endpoint)"},
    "api_key": {"type": "string"}
  }
}
```

**Done event:**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://bank-speech-ai/schemas/synthesize-done.json",
  "title": "StreamingSynthesizeDone",
  "type": "object",
  "required": ["type", "chunks"],
  "properties": {
    "type": {"const": "done"},
    "chunks": {"type": "integer", "minimum": 0}
  }
}
```

### 3.5 Worked example (Python, `websockets`)

```python
import asyncio, json, websockets

async def stream_tts(text, url="ws://localhost:8000/v1/ws/synthesize", api_key=None):
    payload = {"text": text, "speed": 1.0}
    if api_key:
        payload["api_key"] = api_key
    wav_chunks = []
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(payload))
        async for message in ws:
            if isinstance(message, str):
                done = json.loads(message)
                assert done["type"] == "done" and done["chunks"] == len(wav_chunks)
                break
            wav_chunks.append(message)        # each is a complete WAV blob
    return wav_chunks

asyncio.run(stream_tts("Hello. Goodbye."))
```

---

## 4. Errors

| Condition | Channel | Close code / behavior |
|---|---|---|
| Missing/wrong `api_key` (first JSON message) | both | `1008` |
| First transcribe message is not JSON, or config invalid | transcribe | `1011` |
| Engine load/transcription/synthesis failure | both | `1011` (+ `speech_errors_total` counter, logged with `request_id`) |
| Empty `text` on synthesize | synthesize | `1011` (server raises before streaming) |
| Client disconnects mid-stream | both | best-effort flush of pending finals; server closes |

After an error close the connection is gone — clients should reconnect (with
backoff) for resilience.

---

## 5. Observability

Per connection, the server maintains the gauge `speech_ws_active{kind}`
(`kind` = `transcribe` | `synthesize`) — incremented on accept, decremented on
close. All session logs carry the connection's `request_id`. No per-message
metrics are emitted (avoiding cardinality blowup).

---

## 6. OpenAPI integration

FastAPI does **not** include WebSocket routes in its generated OpenAPI schema.
This platform injects them (module `speechai.api.ws_openapi`, wired in
`speechai.api.app.create_app`) so the endpoints are visible in:

- **Swagger UI** — `GET /docs` shows both paths with summaries, descriptions
  and the `x-websocket: true` vendor extension.
- **Raw schema** — `GET /openapi.json` contains:

  - `paths["/v1/ws/transcribe"]` and `paths["/v1/ws/synthesize"]`, each a
    `get`-shaped operation carrying:
    - `x-websocket: true` — marks it as a WebSocket endpoint (not an HTTP GET),
    - `x-wire-protocol: "docs/ws-protocol.md"` — pointer to this spec,
    - `summary` / `description` — the protocol in prose.
  - `x-websocket-spec: "docs/ws-protocol.md"` — top-level pointer to this file.

The REST reference (request/response schemas for `/v1/transcribe`,
`/v1/synthesize`, `/v1/jobs/*`) remains the standard OpenAPI content generated
from the Pydantic models in `speechai.api.schemas`.

> **Compatibility note:** the `x-websocket` vendor extension is non-standard by
> design (OpenAPI 3.x has no native WebSocket support). Consumers must read the
> extension explicitly; the canonical contract is this document.
