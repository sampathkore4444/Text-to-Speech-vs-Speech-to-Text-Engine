# Architecture & design decisions

## 1. High-level flow

Two execution paths serve different latency/SLA profiles:

1. **Synchronous path** (`POST /v1/transcribe`, `POST /v1/synthesize`): low-latency,
   single-request processing intended for small files and IVR interactions. The API
   process runs the engine call in a worker thread (`asyncio.to_thread`) so the event
   loop stays responsive.
2. **Asynchronous path** (`POST /v1/jobs/*`): long-running batch workloads (recorded
   calls, corpus transcription, bulk IVR message generation). Jobs are enqueued,
   picked up by **N worker replicas** sharing a Redis queue, and tracked through a
   state machine: `queued → running → succeeded | failed`.

3. **Streaming paths** (`/v1/ws/transcribe`, `/v1/ws/synthesize`): real-time
   interaction. ASR is *segment-based*: a streaming VAD splits the PCM stream into
   utterances; each completed utterance is transcribed and emitted as a final event,
   with periodic partial hypotheses in between. TTS is *sentence-chunked* so audio
   playback starts before the full message is generated.

## 2. Why these engines?

| Layer | Choice | Rationale |
|---|---|---|
| ASR | **faster-whisper** (CTranslate2) | Whisper accuracy with int8 CPU inference; RTF < 0.5 for `base`/`small` on commodity hardware; no GPU required; audio stays on-prem |
| VAD | **WebRTC VAD** (energy fallback) | No heavy ML stack, tuned for telephony, deterministic frame behavior; Silero VAD documented as an upgrade path |
| TTS | **Piper** (ONNX) | High-quality neural TTS on CPU, per-sentence streaming, tiny runtime vs. autoregressive torch models; voice cloning (XTTS) documented as upgrade |
| Queue | **Redis** (memory fallback) | Mature, observable, BLPOP fair scheduling; memory backend for single-process dev/tests |
| Metrics | **Prometheus** | Standard for SLO alerting; gauges/histograms for WER, CER, RTF, latency, queue depth |

The engine *protocols* (`STTEngine`, `TTSEngine`) make swapping backends a factory
change (`build_stt_engine` / `build_tts_engine`) — e.g. Azure Speech or a fine-tuned
Whisper variant.

## 3. Key modules

### `core/`
- **config**: single YAML + `SPEECHAI_*` env overlay (`__` = section separator).
- **logging**: JSON logs with `request_id` / `job_id` correlation via contextvars;
  every log line is machine-parseable for SIEM ingestion.
- **metrics**: shared Prometheus registry — one source of truth for SLIs.

### `audio/vad.py`
`StreamingVAD` implements hysteresis segmentation:
- speech must persist ≥ `min_speech_ms` to open an utterance (no click-triggered starts),
- silence ≥ `min_silence_ms` closes it (with trailing-silence trimming),
- a hard `max_utterance_ms` cap bounds worst-case transcription latency.

### `stt/streaming.py`
Utterance boundaries come from VAD, not the model — the classic production pattern for
local Whisper streaming. Partials are *not* redacted (avoids flickering masks), finals
are cleaned and redacted.

### `redaction/pii.py`
Ordered pattern application with Luhn validation for card numbers and currency-aware
guards against phone-number false positives (e.g. `$1,250.50` is untouched). Errs on
the side of masking — a false positive hides a few digits, a false negative leaks a PAN.

### `tts/textnorm.py`
Bank-specific text normalization applied **after** redaction so sensitive values are
replaced with "redacted" before digit expansion — a card number is *never* spoken.
Supports currency, dates (DMY/ISO), times, percentages, long reference numbers
(digit-by-digit), and thousands-grouped amounts.

### `pipeline/`
`JobQueue` protocol with two backends. The memory backend is deliberately
loop-agnostic (threading lock + deque) so the API and worker can share it in-process
for tests and single-process deployments.

### `eval/`
`run_evaluation` transcribes a manifest, computes per-utterance WER/CER/RTF/latency,
produces an aggregate report, exports Prometheus gauges, and enforces regression
gates (`--gate`).

## 4. Reliability & observability

- **Health/readiness**: `/health` reports model load state + queue depth; Docker
  healthcheck curls it.
- **SLO alerts**: `deploy/prometheus/alerts.yml` covers WER regression, RTF
  degradation, API down, job failures, queue backlog, model unloaded, p95 latency.
- **Error taxonomy**: typed errors with stable codes (`model_not_found`,
  `payload_too_large`, ...) mapped to HTTP statuses; retryable flags for the
  worker.
- **Audit**: every job/request is logged with correlation ids; uploads retained.

## 5. Security model

On-prem inference, optional API-key auth (HTTP + WebSocket), upload caps, PII
redaction at the text layer for both directions of the pipeline, and structured
audit logs. See `docs/production-checklist.md` for the remaining hardening items
(OIDC, SIEM export, encryption at rest, pen-testing).

## 6. Testing strategy

- **Unit tests** — VAD segmentation, redaction (incl. Luhn + false-positive guards),
  text normalization, audio IO, eval math.
- **Contract tests** — FastAPI `TestClient` with deterministic stub engines: REST,
  WebSocket streaming, auth, and a full API→queue→worker→result round trip with a
  real worker thread.
- **Model tests** — marked manual/CI-optional because they download weights
  (`speechai evaluate` is the gate).
