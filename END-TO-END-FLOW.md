# Bank Speech AI Platform — End-to-End Flow

> **Complete walkthrough of the platform:** what it is, how every phase works in
> minute detail, how a frontend/client talks to the backend, and exactly how to
> start and stop all servers — with and without Docker.
>
> This document is the "one stop" companion to the codebase. It mirrors
> `README.md` (quickstart), `docs/architecture.md` (design decisions) and
> `configs/config.yaml` (tunables), and adds the full request lifecycle, wire
> protocols, and operations runbook.

---

## Table of contents

1. [Platform overview](#1-platform-overview)
2. [System architecture](#2-system-architecture)
3. [Component inventory](#3-component-inventory)
4. [Configuration deep dive](#4-configuration-deep-dive)
5. [The end-to-end phases](#5-the-end-to-end-phases)
   - [Request lifecycle sequence diagrams](#50-request-lifecycle-sequence-diagrams)
   - [Phase 0 — Provisioning](#phase-0--provisioning)
   - [Phase 1 — Client request lifecycle (API entry)](#phase-1--client-request-lifecycle-api-entry)
   - [Phase 2 — STT (speech-to-text) pipeline](#phase-2--stt-speech-to-text-pipeline)
   - [Phase 3 — TTS (text-to-speech) pipeline](#phase-3--tts-text-to-speech-pipeline)
   - [Phase 4 — Async batch pipeline (jobs + queue + workers)](#phase-4--async-batch-pipeline-jobs--queue--workers)
   - [Phase 5 — Real-time streaming (WebSocket)](#phase-5--real-time-streaming-websocket)
   - [Phase 6 — Banking PII redaction](#phase-6--banking-pii-redaction)
   - [Phase 7 — Evaluation harness (WER / CER / RTF / latency)](#phase-7--evaluation-harness-wer--cer--rtf--latency)
   - [Phase 8 — Model fine-tuning (LoRA)](#phase-8--model-fine-tuning-lora)
   - [Phase 9 — Observability (metrics, logs, health, alerts)](#phase-9--observability-metrics-logs-health-alerts)
6. [How the frontend communicates with the backend](#6-how-the-frontend-communicates-with-the-backend)
7. [Starting & stopping the servers](#7-starting--stopping-the-servers)
   - [Without Docker (bare-metal / local dev)](#71-without-docker-bare-metal--local-dev)
   - [With Docker (production-like stack)](#72-with-docker-production-like-stack)
8. [Troubleshooting & operational notes](#8-troubleshooting--operational-notes)
9. [Reference: endpoints, schemas, error codes, env vars](#9-reference-endpoints-schemas-error-codes-env-vars)

---

## 1. Platform overview

**Bank Speech AI** is a production-grade, on-premise **speech AI platform (STT + TTS)**
built for banking. Everything runs locally on CPU with open-source models, so
customer audio never leaves the bank's infrastructure.

What the platform provides, feature by feature:

| Capability | How it is delivered |
|---|---|
| **Speech-to-text (STT / ASR)** | `faster-whisper` (CTranslate2, int8 on CPU) with optional VAD filtering |
| **Text-to-speech (TTS)** | `Piper` (ONNX / onnxruntime) neural voices |
| **Real-time / near-real-time speech** | WebSocket endpoints: VAD-gated utterance ASR + sentence-chunked TTS |
| **Batch processing** | Async job queue (`queued → running → succeeded/failed`) with Redis (or in-memory) backend and horizontally scalable workers |
| **Banking-grade safety** | PII redaction (cards with Luhn validation, accounts, IFSC, Aadhaar, PAN, SSN, phones, emails) on **both** ASR output and TTS input |
| **Quality metrics** | Evaluation harness (`speechai evaluate`): WER, CER, RTF, latency + Prometheus SLIs + SLO alerts |
| **Fine-tuning** | LoRA fine-tuning of Whisper on bank-domain audio, exported to CTranslate2 and hot-swapped via config |
| **Observability** | JSON structured logs with request/job correlation ids, Prometheus `/metrics`, typed error taxonomy |
| **Security** | On-prem inference, optional `X-API-Key` auth (HTTP + WebSocket), upload caps, retained uploads |

**Technology stack:** Python ≥ 3.10 · FastAPI · Uvicorn · Pydantic v2 ·
faster-whisper (CTranslate2) · Piper (onnxruntime) · WebRTC/energy VAD ·
Redis (queue) · Prometheus · jiwer · soundfile · numpy.

---

## 2. System architecture

```
                        ┌────────────────────────────────────────────────┐
                        │                    Clients                     │
                        │  REST (sync + async)   ·   WebSocket streaming │
                        │  (browser/SPA, mobile, IVR, scripts/demo, CLI) │
                        └───────────────┬────────────────────────────────┘
                                        │ HTTP/WS + X-API-Key
                      ┌─────────────────▼──────────────────┐
                      │         API service (FastAPI)       │
                      │  /v1/transcribe  /v1/synthesize     │
                      │  /v1/jobs/*      /v1/ws/*           │
                      │  Observability + auth middleware    │
                      └───────┬──────────────────┬──────────┘
                              │ sync path        │ async path
                              ▼                  ▼
                    ┌─────────────────┐   ┌──────────────────┐
                    │   BatchPipeline │   │  Job queue       │
                    │ (STT/TTS engine)│   │  memory / Redis  │
                    └────────┬────────┘   └────────┬─────────┘
                             │                     │
                             ▼                     ▼
                  ┌────────────────────┐   ┌──────────────────┐
                  │   STT engines      │   │  Batch worker(s) │
                  │   faster-whisper   │   │  (scale-out)     │
                  │   + VAD + redact   │   └────────┬─────────┘
                  │   TTS engines      │            │
                  │   Piper + textnorm │◄───────────┘
                  └────────┬───────────┘
                           │
        ┌──────────────────┼────────────────────┐
        ▼                  ▼                    ▼
   Prometheus         PII redaction        Evaluation
   metrics/SLOs       (bank security)      WER/CER/RTF reports
```

**Four execution paths** (this is the single most important concept):

1. **Synchronous REST** (`POST /v1/transcribe`, `POST /v1/synthesize`) — one
   request, processed inline (in a worker thread via `asyncio.to_thread` so the
   event loop stays responsive). For small files / IVR interactions.
2. **Async batch REST** (`POST /v1/jobs/*` → `GET /v1/jobs/{id}`) — long-running
   work (recorded calls, corpus transcription). Enqueued, picked up by N worker
   replicas, tracked through a state machine.
3. **Streaming WebSocket ASR** (`/v1/ws/transcribe`) — client streams raw PCM16;
   server VAD-splits into utterances and returns `partial` + `final` events.
4. **Streaming WebSocket TTS** (`/v1/ws/synthesize`) — client sends text; server
   returns one WAV blob per sentence.

**Layering** (Python packages under `src/speechai/`):

| Layer | Responsibility |
|---|---|
| `speechai.core` | config (YAML + env overlay), structured logging, Prometheus metrics, typed errors, timing |
| `speechai.audio` | `AudioBuffer` (float32 mono), resampling (soxr/linear), WAV/PCM codecs, VAD |
| `speechai.stt` | engine protocol + factory, faster-whisper engine, post-processing, streaming transcriber |
| `speechai.tts` | engine protocol + factory, Piper engine, bank text normalization, streaming synthesizer |
| `speechai.redaction` | banking PII detection + masking (Luhn-validated) |
| `speechai.pipeline` | `Job` model + state machine, queue backends (memory/Redis), batch orchestrator |
| `speechai.eval` | dataset loading (JSONL/CSV/dir), WER/CER/RTF/latency, reports, regression gates |
| `speechai.finetune` | LoRA fine-tuning of Whisper + CTranslate2 export (`speechai-finetune`) |
| `speechai.api` | FastAPI app factory, REST routes, WebSockets, middleware, Pydantic schemas |
| `speechai.workers` | batch worker process (`python -m speechai.workers.batch_worker`) |
| `speechai.cli` | `speechai` command line (transcribe/synthesize/evaluate/models) |

---

## 3. Component inventory

Every source file and what it does:

```
src/speechai/
  core/
    config.py      Pydantic Settings: YAML + SPEECHAI_* env overlay (__ = section sep)
    errors.py      SpeechAIError hierarchy with stable codes + HTTP status mapping
    logging.py     JSON formatter; request_id / job_id correlation via contextvars
    metrics.py     Shared Prometheus registry (counters/gauges/histograms)
    timing.py      Stopwatch + compute_rtf() helper
  audio/
    io.py          AudioBuffer (mono float32), load_audio, resample, to_asr_audio,
                   pcm16_bytes, write_wav, from_wav_bytes, from_pcm16
    vad.py         WebRTCVAD, EnergyVAD, build_vad factory, StreamingVAD segmenter
  stt/
    base.py        STTOptions, Segment, TranscriptionResult, STTEngine protocol, factory
    whisper_engine.py  faster-whisper engine (CTranslate2, int8 CPU / fp16 GPU)
    postprocess.py TextPostProcessor: clean_text + PII redaction of ASR output
    streaming.py   StreamingTranscriber: VAD-gated utterance transcription
  tts/
    base.py        TTSOptions, SynthesisResult, TTSEngine protocol, factory
    piper_engine.py     Piper engine (onnxruntime); multi-API-generation support
    textnorm.py         TextNormalizer: redact + expand numbers/dates/currency/...
    streaming.py        StreamingSynthesizer: per-sentence WAV chunks
  redaction/
    pii.py         Redactor + RedactionPolicy; patterns + Luhn + phone guards
  pipeline/
    jobs.py        Job dataclass + state machine + new_job_id()
    queue.py       JobQueue protocol; MemoryJobQueue; RedisJobQueue; build_queue()
    batch.py       BatchPipeline: submit/get/run jobs; sync transcribe/synthesize
  eval/
    loader.py      load_manifest: JSONL | CSV | directory of wav+txt pairs
    metrics.py     jiwer WER/CER, RTF, latency, aggregates, EvaluationReport
    runner.py      run_evaluation, run_from_manifest, assert_within_tolerance
  finetune/
    dataset.py     WhisperDataset (log-mel features + labels) on platform manifests
    train.py       speechai-finetune CLI: LoRA train, WER before/after, CT2 export
  api/
    app.py         create_app() factory; lifespan wiring; /health, /metrics; handlers
    routes.py      All REST endpoints under /v1
    ws.py          WebSocket /v1/ws/transcribe + /v1/ws/synthesize
    schemas.py     Pydantic request/response models
    middleware.py  ObservabilityMiddleware + APIKeyMiddleware
  workers/
    batch_worker.py  Worker loop: dequeue -> run_job -> update
  cli/
    main.py        speechai CLI (argparse + rich tables)
scripts/
  download_models.py  Piper voice download + optional Whisper cache warm
  make_sample_audio.py  TTS -> WAV -> manifest (demo loop + eval data)
  streaming_demo.py    Reference WebSocket client (mic/file -> ASR, text -> TTS)
deploy/prometheus/
  prometheus.yml   Scrape config (api:8000)
  alerts.yml       SLO alert rules (WER, RTF, down, job failures, backlog, ...)
```

---

## 4. Configuration deep dive

### 4.1 The single source of truth

Everything is configured by **one YAML file** — `configs/config.yaml` — overlaid
by **environment variables** at runtime. Env vars use `__` as the section
separator:

```bash
SPEECHAI_CONFIG=/path/to/config.yaml      # which YAML file to load
SPEECHAI_API__PORT=9000                   # overrides api.port
SPEECHAI_STT__MODEL_SIZE=small            # overrides stt.model_size
SPEECHAI_QUEUE__BACKEND=redis             # overrides queue.backend
SPEECHAI_QUEUE__REDIS_URL=redis://redis:6379/0
SPEECHAI_STORAGE__DATA_DIR=/data
```

Scalar coercion is automatic: `"true"/"1"/"yes"/"on"` → `True`, `"false"/"0"` →
`False`, `"null"/"none"/""` → `None`, JSON values parse as their type, anything
else stays a string (`speechai.core.config._coerce_scalar`).

**Load order** (`Settings.load()`): read YAML (if the file exists) → deep-copy →
apply every `SPEECHAI_*` env var over the tree → validate with Pydantic.

### 4.2 Every config key

| Section | Key | Default | Meaning |
|---|---|---|---|
| `service` | `name` | `bank-speech-ai` | Log/telemetry service name |
| | `environment` | `development` | `development` \| `staging` \| `production` |
| | `log_level` | `INFO` | Root logger level |
| | `log_format` | `json` | `json` (structured) or `text` |
| `api` | `host` | `0.0.0.0` | Bind address (uvicorn uses this) |
| | `port` | `8000` | HTTP/WS port |
| | `api_key` | `""` | Empty = auth **disabled**; set a value to require `X-API-Key` |
| | `max_upload_mb` | `50` | Upload cap → HTTP 413 above it |
| | `timeout_seconds` | `300` | Request timeout budget (declared; not yet enforced by any code path) |
| `storage` | `data_dir` | `data` | Root for uploads, results, models |
| | `result_ttl_seconds` | `86400` | How long jobs/artifacts live (1 day) |
| | `max_results` | `1000` | Hard cap on retained jobs |
| `stt` | `engine` | `whisper` | Engine name (factory key) |
| | `model_size` | `base` | `tiny`→`large-v3` or an HF id |
| | `model_path` | `""` | Local CTranslate2 dir (fine-tuned export); **takes precedence** over `model_size` |
| | `device` | `auto` | `auto` \| `cpu` \| `cuda` |
| | `compute_type` | `auto` | `auto` → `int8` on CPU, `float16` on CUDA |
| | `beam_size` | `5` | Whisper beam search width |
| | `language` | `null` | `null` = auto-detect; pin e.g. `en` for lower latency |
| | `vad_filter` | `true` | faster-whisper built-in VAD |
| | `min_silence_ms` | `500` | Silence that closes a streaming utterance |
| | `max_segment_ms` | `12000` | Hard cap on one streaming utterance |
| | `partial_interval_ms` | `2500` | Cadence of `partial` (non-final) events |
| `tts` | `engine` | `piper` | Engine name (factory key) |
| | `voice` | `en_US-lessac-medium` | Voice name; must exist under `data/models/voices/` |
| | `model_path` | `""` | Explicit `.onnx` path; empty = look under voices dir |
| | `sample_rate` | `22050` | Output sample rate |
| | `default_speed` | `1.0` | Default speaking rate (0.5–2.0) |
| `vad` | `backend` | `auto` | `webrtc` \| `energy` \| `auto` (webrtc first, energy fallback) |
| | `frame_ms` | `30` | VAD frame length (WebRTC: 10/20/30) |
| | `aggressiveness` | `2` | WebRTC 0..3 (higher = more aggressive) |
| | `energy_threshold_db` | `-35.0` | Energy fallback threshold |
| | `min_speech_ms` | `250` | Speech must persist this long to open an utterance |
| | `min_silence_ms` | `400` | Silence closes an utterance (VAD layer) |
| `queue` | `backend` | `memory` | `memory` \| `redis` (compose default: redis) |
| | `redis_url` | `redis://localhost:6379/0` | Redis connection |
| | `poll_interval_seconds` | `1.0` | Worker BLPOP timeout / idle poll |
| | `job_timeout_seconds` | `900` | Job timeout budget (declared; not yet enforced — the worker runs each job to completion) |
| `redaction` | `enabled` | `true` | Master switch |
| | `mode` | `mask` | `mask` (keep last 4) \| `redact` (`[REDACTED]`) \| `none` |
| | `mask_keep_last` | `4` | Digits kept visible in mask mode |
| | `patterns.*` | all `true` | Toggle each PII type: card, account, ifsc, aadhaar, pan, ssn, phone, email |
| `eval` | `default_wer_tolerance` | `0.10` | `--gate` WER ceiling (10%) |
| | `default_rtf_tolerance` | `0.50` | `--gate` RTF ceiling |
| | `report_dir` | `data/eval` | Where eval JSON reports go |

### 4.3 Derived data directories

Resolved lazily from `storage.data_dir` (in Docker: `/data` via
`SPEECHAI_STORAGE__DATA_DIR`):

| Path | Contents |
|---|---|
| `data/uploads/` | Incoming audio files, named `<uuid4hex>.wav`, retained for audit/compliance |
| `data/results/` | Batch artifacts: `<job_id>.wav` for synthesis jobs |
| `data/models/` | Model root |
| `data/models/voices/` | Piper voices (`<voice>.onnx` + `.onnx.json`) |
| `data/eval/` | Evaluation reports |
| `data/samples/` | Generated sample audio + reference transcripts \* |

\* Created by `scripts/make_sample_audio.py`; **not** auto-created by
`ensure_dirs()`. The other five directories above are auto-created at startup.

`Settings.ensure_dirs()` creates all of them on startup.

---

## 5. The end-to-end phases

### 5.0 Request lifecycle sequence diagrams

The four execution paths, drawn against the actual code. Lifelines are the
actors in `src/speechai/api/` and `src/speechai/pipeline/`; numbered steps map
1:1 to the handler/pipeline logic. Section 6 covers the wire-level message
formats; these diagrams show *who calls whom, when*.

#### 5.0.1 Synchronous STT — POST /v1/transcribe

```text
 Client            Auth MW            Observability MW      Route /v1/transcribe    BatchPipeline           STT Engine
   │                  │                     │                     │                   │                        │
   │ 1) POST /v1/transcribe (multipart: file, language, redact) + X-API-Key (if auth on)
   ├─────────────────►│                     │                     │                   │                        │
   │                  │ 2) key valid?       │                     │                   │                        │
   │                  │    wrong → 401 {'error':{'code':'unauthorized',...}}
   │                  ├────────────────────►│                     │                   │                        │
   │                  │                     │ 3) request_id = uuid4().hex[:12]
   │                  │                     │    set_request_id(); start perf timer
   │                  │                     ├────────────────────►│                   │                        │
   │                  │                     │                     │ 4) await file.read(); _validate_upload()
   │                  │                     │                     │    empty → 400 | > max_upload_mb → 413
   │                  │                     │                     │ 5) write data/uploads/<uuid>.wav
   │                  │                     │                     ├──────────────────►│                        │
   │                  │                     │                     │    asyncio.to_thread(transcribe_sync)
   │                  │                     │                     │                   │ 6) lazy load()         │
   │                  │                     │                     │                   │    WhisperModel(...)    │
   │                  │                     │                     │                   │ 7) transcribe(audio)   │
   │                  │                     │                     │                   ├───────────────────────►│
   │                  │                     │                     │                   │                        │ 8) beam search
   │                  │                     │                     │                   │◄───────────────────────│
   │                  │                     │                     │                   │  TranscriptionResult  │
   │                  │                     │                     │                   │ 9) postprocess: clean  │
   │                  │                     │                     │                   │    + PII redaction     │
   │                  │                     │                     │                   │ 10) record stt_* metrics
   │                  │                     │◄────────────────────│                   │                        │
   │                  │                     │  TranscribeResponse │                   │                        │
   │                  │                     │ 11) x-request-id header; record http_* metrics
   │                  │◄────────────────────│                     │                   │                        │
   │◄─────────────────│                     │                     │                   │                        │
   │ 12) 200 JSON: text, language, engine, segments[], redacted, redactions[],
   │     metrics{latency_seconds, engine_seconds, rtf, audio_duration_seconds, confidence}, request_id
```

#### 5.0.2 Synchronous TTS — POST /v1/synthesize

```text
 Client            API service            BatchPipeline         TextNormalizer +       TTS Engine (Piper)
   │               (middleware + route)       │                  PII Redactor                │
   │                    │                    │                       │                     │
   │ 1) POST /v1/synthesize {text, voice?, speed} (or ?stream=true)
   ├───────────────────►│                    │                       │                     │
   │                    │ 2) Pydantic validate (422 on empty text)   │                     │
   │                    ├───────────────────►│                       │                     │
   │                    │                    │ 3) normalize(text)    │                     │
   │                    │                    ├──────────────────────►│                     │
   │                    │                    │                       │ 4) redact PII; masked runs → 'redacted'
   │                    │                    │                       │ 5) expand dates / times / % / $ / numbers
   │                    │                    │◄──────────────────────│ NormalizedText     │
   │                    │                    │ 6) synthesize(text, speed)                  │
   │                    │                    ├───────────────────────────────────────────►│
   │                    │                    │                                            │ 7) lazy load PiperVoice
   │                    │                    │                                            │ 8) synthesize → WAV
   │                    │                    │◄───────────────────────────────────────────│ SynthesisResult
   │                    │                    │ 9) record tts_* metrics                     │
   │                    │◄───────────────────│ 10) audio/wav bytes (or chunk stream)       │
   │◄───────────────────│                    │                       │                     │
   │ 11) 200 audio/wav; ?stream=true → one WAV blob per sentence
```

#### 5.0.3 Async batch — POST /v1/jobs/transcribe → poll → artifact

```text
 Client            API service           Job queue             Worker (replica)      Pipeline              Engine
   │                    │               (memory / Redis)           │                    │                     │
   │ 1) POST /v1/jobs/transcribe (multipart upload)
   ├───────────────────►│                    │                       │                    │                     │
   │                    │ 2) validate + save data/uploads/<uuid>.wav│                    │                     │
   │                    │ 3) submit_transcribe(job)                 │                    │                     │
   │                    ├───────────────────►│                      │                    │                     │
   │                    │                    │ enqueue → status=queued                    │                     │
   │◄───────────────────│                    │                      │                    │                     │
   │ 4) 202 {job_id, status='queued', url}   │                      │                    │                     │
   │                    │                    │ 5) BLPOP / dequeue   │                    │                     │
   │                    │                    ├─────────────────────►│                    │                     │
   │                    │                    │                      │ 6) run_job(job)    │                     │
   │                    │                    │◄─────────────────────│ status=running; queue.update
   │                    │                    │                      ├───────────────────►│                     │
   │                    │                    │                      │ 7) transcribe_sync / synthesize_sync
   │                    │                    │                      │                    ├────────────────────►│
   │                    │                    │                      │                    │ 8) engine call      │
   │                    │                    │                      │◄───────────────────│ result              │
   │                    │                    │                      │ 9) mark_succeeded / mark_failed; update
   │                    │                    │◄─────────────────────│                      │                     │
   │ 10) GET /v1/jobs/{id} (poll)            │                      │                    │                     │
   ├───────────────────►│                    │                      │                    │                     │
   │                    │ get(job_id)        │                      │                    │                     │
   │                    │◄───────────────────│                      │                    │                     │
   │◄───────────────────│ 11) 200 JobStatusResponse (status, result, metrics, audio_url)
   │ 12) GET /v1/jobs/{id}/audio → WAV (synthesis jobs only)
   │ 13) DELETE /v1/jobs/{id} → 204
```

#### 5.0.4 WebSocket streaming ASR — /v1/ws/transcribe

```text
 Client            WS route              StreamingTranscriber     StreamingVAD           STT Engine
   │               (/v1/ws/transcribe)     (per connection)      (webrtc / energy)       (Whisper)
   │                    │                        │                     │                       │
   │ 1) WS handshake    │                        │                     │                       │
   ├───────────────────►│ accept(); speech_ws_active{transcribe}++   │                     │
   │ 2) JSON config {sample_rate, language, api_key?, partial_interval_ms}
   ├───────────────────►│                        │                     │                       │
   │                    │ build StreamingTranscriber (VAD + engine)   │                       │
   │ 3) binary PCM16 chunks (any size)           │                     │                       │
   ├───────────────────►│ feed(chunk)            │                     │                       │
   │                    ├───────────────────────►│ push(samples)       │                       │
   │                    │                        ├────────────────────►│ 4) 30 ms frame VAD    │
   │                    │                        │◄────────────────────│ utterance completed   │
   │                    │                        │ 5) transcribe (worker thread)              │
   │                    │◄───────────────────────│ 6) partial event (every partial_interval_ms)
   │◄───────────────────│ {'type':'partial','text':...}                │                       │
   │                    │◄───────────────────────│ 7) final event (cleaned + redacted)       │
   │◄───────────────────│ {'type':'final','text':...,'segments':[...]} │                       │
   │ 8) {'action':'stop'} (or disconnect)        │                     │                       │
   ├───────────────────►│ finish() flush         │                     │                       │
   │                    │◄───────────────────────│ 9) last final event │                       │
   │◄───────────────────│                        │                     │                       │
   │                    │ 10) close(); speech_ws_active--              │                       │
```

#### 5.0.5 WebSocket streaming TTS — /v1/ws/synthesize

```text
 Client            WS route              StreamingSynthesizer     TTS Engine (Piper)
   │               (/v1/ws/synthesize)     (per connection)
   │                    │                        │                       │
   │ 1) WS handshake    │                        │                       │
   ├───────────────────►│ accept(); speech_ws_active{synthesize}++     │
   │ 2) JSON {text, speed, api_key?}            │                       │
   ├───────────────────►│ synthesize(text)       │                       │
   │                    ├───────────────────────►│ split_sentences()     │
   │                    │                        ├──────────────────────►│ 3) synthesize(sentence)
   │                    │◄───────────────────────│ WAV bytes             │
   │◄───────────────────│ 4) binary WAV frame (client starts playback)   │
   │                    │                        │ (repeat per sentence) │
   │◄───────────────────│ 5) {'type':'done','chunks':N}                  │
   │                    │ 6) close(); speech_ws_active--                 │
```

> Each step above is the real call chain. For the exact JSON/binary framing of
> every arrow, see [Section 6 — frontend ↔ backend communication](#6-how-the-frontend-communicates-with-the-backend).

### Phase 0 — Provisioning

Happens once per environment, before any traffic.

1. **Install the package** (Python ≥ 3.10):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
   pip install -e ".[dev,engines]"
   ```

   - `[dev]` adds pytest + ruff.
   - `[engines]` adds the inference stack: `faster-whisper`, `piper-tts`,
     `webrtcvad-wheels`, `soxr`.
   - `[finetune]` (optional) adds torch/transformers/peft for LoRA training.

2. **Fetch models** (`scripts/download_models.py`):

   ```bash
   python scripts/download_models.py                          # Piper default voice
   python scripts/download_models.py --piper-voice en_US-amy-medium
   python scripts/download_models.py --whisper-size base       # warm the Whisper cache too
   ```

   - **Piper voices must be downloaded explicitly** into `data/models/voices/`
     (from `rhasspy/piper-voices` on HuggingFace).
   - **Whisper downloads automatically** from the HF Hub on first load; the
     `--whisper-size` flag just warms the cache. A local converted CTranslate2
     dir (from fine-tuning) can be used instead via `stt.model_path`.

3. **(Optional) Generate bank-domain sample audio** — a full TTS → WAV → STT demo
   loop plus an evaluation manifest:

   ```bash
   python scripts/make_sample_audio.py
   # writes data/samples/*.wav + *.txt and data/manifest.jsonl
   ```

4. **Sanity-check the stack**:

   ```bash
   speechai models                     # prints engine/model configuration
   pytest                              # runs the full unit + contract test suite
   ```

### Phase 1 — Client request lifecycle (API entry)

Every request (HTTP or WebSocket) flows through the same startup-wired
application. At boot, `create_app()`:

1. Loads `Settings`, calls `ensure_dirs()`.
2. Registers the **lifespan** context which, on startup:
   - `setup_logging(...)` — replaces root handlers with JSON (or text) formatting.
   - `build_queue(settings)` — memory or Redis queue.
   - Builds the `Redactor` from the redaction policy.
   - Builds a `BatchPipeline(settings, queue, redactor)` and stashes everything on
     `app.state` (`settings`, `queue`, `redactor`, `pipeline`).
   - On shutdown, closes the queue (`await queue.close()`).
3. Registers middleware **in reverse order of call** (last added runs first):
   - `APIKeyMiddleware` (only if `api.api_key` is set) — outermost.
   - `ObservabilityMiddleware` — sets `x-request-id`, times the request, records
     `http_requests_total` + `http_latency_seconds` (path label is templated to
     keep cardinality low: `/v1/jobs/abcd…` → `/v1/jobs/{id}`).
4. Mounts the REST router (`/v1`, tags `speech`), the WebSocket router, and
   exception handlers for `SpeechAIError` (→ its mapped HTTP status + JSON
   `{"error": {...}}`) and generic `Exception` (→ 500 `internal_error`).
5. Registers `/health` and `/metrics` (outside `/v1`).

**Request path, step by step (sync STT example):**

1. Client uploads audio via `multipart/form-data` to `POST /v1/transcribe`.
2. Middleware runs: auth check (if enabled) → request-id assignment → timing.
3. Route handler:
   - Reads the whole file (`await file.read()`).
   - `_validate_upload`: rejects empty files (400 `validation_error`) and files
     over `max_upload_mb` (413 `payload_too_large`).
   - Persists bytes to `data/uploads/<uuid>.wav` (retained for audit).
   - Calls `pipeline.transcribe_sync(...)` inside `asyncio.to_thread` so the
     blocking CPU work never stalls the event loop.
   - Returns a `TranscribeResponse` with the `x-request-id` echoed.
4. Errors bubble up to the exception handlers, producing a stable JSON error
   envelope and incrementing `speech_errors_total`.

### Phase 2 — STT (speech-to-text) pipeline

Everything below happens inside `BatchPipeline.transcribe_sync` (sync REST) or
`_execute_transcribe` (batch jobs). The CLI mirrors the same steps.

1. **Audio decode** — `load_audio(path)` uses `soundfile` to decode any
   supported format (wav/flac/mp3/ogg/…) into an `AudioBuffer`:
   - Mono **float32** samples in `[-1, 1]` (stereo is down-mixed by averaging).
   - Errors → 400 `audio_format_error`; empty audio rejected.
2. **Resample to ASR format** — `to_asr_audio()` resamples to **16 kHz mono
   float32** using `soxr` when installed (high quality), otherwise a documented
   linear-interpolation fallback.
3. **Inference options** — `STTOptions(language, beam_size, vad_filter)` built
   from per-request overrides and `settings.stt`.
4. **Engine** — `WhisperSTTEngine` (lazy singleton, thread-safe for sequential
   calls, guarded by a lock):
   - `load()`: `WhisperModel(model_ref, device, compute_type)`. `model_ref` is
     `stt.model_path` if set, else `stt.model_size`. On CPU auto → `int8`.
     Load time is recorded to `speech_model_load_seconds`, `speech_model_loaded`
     is set to 1. Model download from HF happens here on first use.
   - `transcribe(audio, opts)`: calls faster-whisper with beam search; converts
     each segment's `avg_logprob` to a confidence via `exp(logprob)`; computes
     `latency_seconds` (wall clock) and `rtf` (latency ÷ audio duration).
5. **Post-processing** — `TextPostProcessor.process(text, redact=...)`:
   - `clean_text`: collapse whitespace → fix spacing around punctuation
     (`hello ,world` → `hello, world`) → sentence capitalization.
   - PII redaction via the shared `Redactor` (see Phase 6). `redact=False`
     (form field) or `--no-redact` (CLI) skips it.
6. **Response assembly** — text, detected language, engine name, per-segment
   `{text, start, end, confidence}`, `redacted` flag, the redaction `findings`
   (`{type, masked}`), and `metrics` (`latency_seconds`, `engine_seconds`, `rtf`,
   `audio_duration_seconds`, `confidence`).
   Segments are **sentence-level**: the engine transcribes with word timestamps
   and rows are regrouped on inter-word pauses (`whisper_engine._group_word_segments`),
   so a multi-sentence file yields one timed line per sentence instead of a
   single whole-file segment (faster-whisper's own VAD only splits on ≥ 2 s
   silences).
7. **Metrics** — increments `stt_requests_total{status,channel}`, observes
   `stt_audio_seconds_total`, `stt_latency_seconds`, `stt_rtf`,
   `stt_confidence`.

### Phase 3 — TTS (text-to-speech) pipeline

Happens inside `BatchPipeline.synthesize_sync` (sync REST), `_execute_synthesize`
(batch jobs), or the CLI.

1. **Validation** — empty/blank text → 400 `validation_error`.
2. **Bank text normalization** — `TextNormalizer.normalize(text)`:
   - **First**, runs PII redaction (Phase 6). Any masked run (e.g. `XXXX XXXX 3456`)
     is then replaced with the literal word **"redacted"** — so a card number is
     *never spoken*, in any mode.
   - **Then**, expands written forms in a fixed order (specific before generic):
     - Dates: `15/08/2026` → "the fifteenth of August twenty twenty six" (also
       `dd-mm-yyyy`, `dd.mm.yyyy`, ISO `yyyy-mm-dd`).
     - Times: `9:30 am` → "nine thirty am"; `9:00 am` → "nine am".
     - Percentages: `3.5%` → "three point five percent".
     - Currency: `$1,250.50` → "one thousand two hundred fifty dollars and fifty
       cents" (symbols `$ € £ ₹` and codes usd/eur/gbp/inr, rupees→paise).
     - Long digit runs (≥ 9 digits, e.g. reference numbers): spoken
       digit-by-digit.
     - Thousands-grouped numbers and remaining plain integers/decimals via
       `num2words`.
   - Returns `NormalizedText(text, redacted, findings)`.
3. **Inference** — `PiperTTSEngine.synthesize(normalized.text, TTSOptions(speed,
   voice))`:
   - `load()`: `PiperVoice.load(<voice>.onnx, config_path=<voice>.onnx.json)`
     from `data/models/voices/` (or `tts.model_path`). Missing file → 503
     `model_not_found` with a hint to run `download_models.py`.
   - Synthesizes to an in-memory WAV buffer. The engine supports all piper-tts
     API generations (`wav_file`/`speed`, `length_scale`, and 1.6+
     `synthesize_wav` + `SynthesisConfig`), trying each in order.
   - Resamples to `tts.sample_rate` (22050) if needed; computes latency + RTF.
4. **Sync path output** — `SynthOutput` wraps the `AudioBuffer`; the route
   returns `Response(content=wav_bytes, media_type="audio/wav",
   Content-Disposition: attachment; filename="speech_<ts>.wav")`.
5. **Metrics** — `tts_requests_total`, `tts_chars_total`, `tts_latency_seconds`,
   `tts_rtf`.

**Streaming mode** (`POST /v1/synthesize?stream=true`) uses
`StreamingSynthesizer`: `split_sentences()` (split on `[.!?]` + whitespace) then
synthesize and yield **one WAV blob per sentence** via `StreamingResponse`,
letting the client start playback before the whole message is generated.

### Phase 4 — Async batch pipeline (jobs + queue + workers)

Designed for long-running workloads that don't fit a synchronous request.

**Job model** (`pipeline/jobs.py`):

- `Job`: `id` (16-hex uuid), `type` (`transcribe` | `synthesize`), `status`,
  timestamps (`created_at`, `started_at`, `finished_at`), `input`, `result`,
  `error`, `metrics`, `attempts`.
- State machine: `queued → running → succeeded | failed` (a `canceled` state
  exists in the enum for future use).

**Queue backends** (`pipeline/queue.py`) — identical `JobQueue` protocol:

- `MemoryJobQueue` — in-process dict + deque, thread-locked, **loop-agnostic** so
  the API and an in-process worker can share it (used by tests and
  single-process deployments). Prunes by TTL (`result_ttl_seconds`) and caps at
  `max_results`.
- `RedisJobQueue` — keys `speech:queue` (list, `RPUSH`/`BLPOP`) and
  `speech:job:{id}` (JSON with TTL). BLPOP gives fair scheduling and supports N
  worker replicas. `redis.asyncio` is imported lazily.

**Submission flow:**

1. `POST /v1/jobs/transcribe` (multipart upload, same validation as sync) or
   `POST /v1/jobs/synthesize` (JSON body).
2. Route creates the `Job`, calls `pipeline.submit_*()` → `queue.enqueue(job)`.
3. Responds **202 Accepted** with `{job_id, status: "queued", url: "/v1/jobs/<id>"}`.

**Worker** (`workers/batch_worker.py`) — `python -m speechai.workers.batch_worker`:

- Builds its own queue + pipeline (same config), then loops:
  `dequeue(timeout=poll_interval_seconds)` → set `job_id` contextvar →
  `pipeline.run_job(job)` → clear `job_id`.
- `run_job`: `mark_started()` → update queue → run in a thread
  (`asyncio.to_thread`) → `mark_succeeded(result)` or `mark_failed(err)` →
  update queue → record `speech_jobs_total{type,status}` and
  `speech_jobs_active`.
- Synthesis jobs additionally persist the WAV artifact to
  `data/results/<job_id>.wav` and expose `audio_url: "/v1/jobs/<id>/audio"`.
- **Horizontal scale-out**: start N replicas sharing the same Redis queue. Each
  worker is single-threaded per replica; queue depth is exported as
  `speech_queue_depth` for autoscaling.

**Client polling:** `GET /v1/jobs/{id}` returns full status
(`JobStatusResponse`) until terminal; `DELETE /v1/jobs/{id}` → 204 removes it
(404 `job_not_found` if absent/expired); `GET /v1/jobs/{id}/audio` streams the
WAV artifact.

> ⚠️ **Operational gotcha:** the **memory queue is per-process**. If you run the
> API and the worker as *separate processes* with the default `memory` backend,
> jobs submitted to the API will never be picked up. Use the **Redis backend**
> (or run both in one process) for any multi-process setup — see Section 7.

### Phase 5 — Real-time streaming (WebSocket)

**Two endpoints**, both under `/v1/ws/`. Each connection increments
`speech_ws_active{kind}`; unexpected failures close with code **1011**, auth
failures with **1008**.

**Fail-fast model loading:** after the first JSON message the handler loads the
engine eagerly (`engine.load()` in a worker thread). A missing/unavailable
model therefore fails **immediately** — the server sends one JSON `error` event
(`{"type": "error", "code": "model_not_found", "message": "..."}`) and then
closes with `1011`. Without this, a lazy load would silently download the model
mid-stream, leaving the client with a hung connection and no feedback.

**ASR: `/v1/ws/transcribe`**

1. Client connects and immediately sends one JSON **config message**:
   `{"sample_rate": 16000, "language": "en"|null, "api_key": "..." (if auth),
   "partial_interval_ms": 2500 (optional)}`.
   - `sample_rate` is the rate of the PCM the client will send (16000
     recommended; other rates are resampled server-side).
2. Client then streams **raw little-endian 16-bit PCM chunks** (mono) as binary
   WebSocket frames. Chunk size is arbitrary — the server feeds them into
   `StreamingVAD` which processes fixed 30 ms frames.
3. `StreamingVAD` (from `audio/vad.py`) segments the stream with hysteresis:
   - Speech must persist ≥ `min_speech_ms` (250 ms) to *open* an utterance
     (no click-triggered starts).
   - Silence ≥ `min_silence_ms` (effective 500 ms from `stt.min_silence_ms`)
     *closes* it, trimming trailing silence.
   - A hard `max_utterance_ms` (12 s) cap bounds worst-case latency.
   - Frame VAD is `build_vad(settings.vad.backend)`: WebRTC preferred
     (`aggressiveness=2`, 30 ms frames), energy fallback (`-35 dB`).
4. Each completed utterance is transcribed in a worker thread and emitted as a
   **`final`** event:
   ```json
   {"type": "final", "text": "...", "start": 0.0, "end": 2.1,
    "utterance_index": 1, "confidence": 0.93, "segments": [{"text": "...", "start": 0.0, "end": 2.1, "confidence": 0.93}]}
   ```
   Finals are cleaned **and redacted** (post-processor).
5. While an utterance is in progress, **`partial`** events are emitted every
   `partial_interval_ms` (default 2.5 s) so the UI can show a live hypothesis:
   ```json
   {"type": "partial", "text": "your account ba", "start": 0.0, "end": 1.2,
    "utterance_index": 1, "confidence": 0.9}
   ```
   Partials are cleaned but **not redacted** (avoids flickering masks).
6. The client ends the stream with `{"action": "stop"}` (or just disconnects);
   the server then flushes any in-progress utterance as one last `final` event
   and closes the socket.

**TTS: `/v1/ws/synthesize`**

1. Client connects and sends one JSON message:
   `{"text": "...", "speed": 1.0 (0.5–2.0), "api_key": "..." (if auth)}`.
2. Server runs `StreamingSynthesizer.synthesize(text)`: splits into sentences
   and synthesizes **each sentence to a WAV blob**, sending it immediately as a
   **binary frame** — the client can start playback chunk-by-chunk.
3. After the last sentence the server sends
   `{"type": "done", "chunks": <n>}` and closes.

The repo's reference client is `scripts/streaming_demo.py` (mic or file → ASR
with live partial/final rendering; text → TTS with per-chunk playback or
`--output out.wav`). It needs `pip install sounddevice websockets` (websockets
already ships with `uvicorn[standard]`).

### Phase 6 — Banking PII redaction

Shared by both directions of the pipeline (`redaction/pii.py`):

- **ASR output** (STT phase 2): transcripts leave the platform clean.
- **TTS input** (TTS phase 3): sensitive values are replaced with the word
  "redacted" so they are *never spoken*.

**Detection** — ordered patterns (order matters to avoid re-matching masked
output), each individually toggleable via `redaction.patterns`:

| Type | Pattern essence | Extra validation |
|---|---|---|
| `card` | 13–19 digits with optional spaces/dashes | **Luhn checksum** — numeric but invalid cards are left alone |
| `aadhaar` | `[2-9]\d{3} \d{4} \d{4}` | — |
| `ssn` | `\d{3}-\d{2}-\d{4}` | — |
| `pan` | `[A-Z]{5}\d{4}[A-Z]` | — |
| `ifsc` | `[A-Z]{4}0[A-Z0-9]{6}` | — |
| `account` | 9–18 digit run | — |
| `phone` | permissive 7–15 digit run with separators | rejects decimals like `1,250.50` and currency-word prefixes (`rs`, `usd`, …) |
| `email` | standard email regex | always `[REDACTED]` |

**Modes** (`redaction.mode`):

- `mask` (default): replace digits with `X`, **keeping the last 4 digits**
  (`4242 4242 4242 4242` → `XXXX XXXX XXXX 4242`); separators preserved.
- `redact`: replace the whole value with `[REDACTED]`.
- `none`: no-op (never for production banking).

**Design rule:** err on the side of redaction — a false positive masks a few
digits; a false negative leaks a PAN or card number.

Every redaction produces a `Finding {pii_type, start, end, masked}` that is
returned in the API response as `redactions: [{type, masked}]` with
`redacted: true`.

### Phase 7 — Evaluation harness (WER / CER / RTF / latency)

Command: `speechai evaluate <manifest> [--language en] [--report path] [--gate]`

1. **Dataset loading** (`eval/loader.py`) — same manifest format as
   fine-tuning: JSONL (`{"audio": ..., "reference": ...}` per line), CSV
   (`audio,reference` columns), or a directory of `<stem>.wav` + `<stem>.txt`
   pairs. Drop in Common Voice or internal call-audio subsets.
2. **Run** (`eval/runner.py`) — for each utterance: load → resample to 16 kHz →
   transcribe → measure wall-clock latency → compute RTF.
3. **Metrics** (`eval/metrics.py`):
   - `WER` / `CER` via `jiwer` (0.0 = perfect).
   - `RTF` = processing seconds ÷ audio seconds (< 1.0 = faster than real time).
   - End-to-end latency per file.
   - Aggregates: **mean / median / p90** per metric + worst-5-by-WER ranking.
4. **Report** — pretty text table in the terminal, JSON export to
   `data/eval/<dataset>-<engine>.json` (or `--report` path). Aggregates are
   pushed to Prometheus gauges `stt_wer{dataset}`, `stt_cer{dataset}`,
   `stt_rtf_mean{dataset}` for regression alerting.
5. **Regression gates** — `--gate` fails the run (exit non-zero) if mean
   WER > `eval.default_wer_tolerance` (0.10) or mean RTF >
   `eval.default_rtf_tolerance` (0.50). CI-friendly.
6. **MOS** — real MOS needs listening panels or NISQA; the harness is structured
   so a MOS column/plugin can be added. Piper voices have published naturalness
   scores.

### Phase 8 — Model fine-tuning (LoRA)

Command: `speechai-finetune --data data/manifest.jsonl --base-model openai/whisper-base --output-dir data/models/finetuned --language en`

(`pip install -e ".[finetune]"` required; on Windows CPU torch:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.)

1. **Data** — reuses the platform manifest loader; audio decoded with the
   platform's own soundfile pipeline (no ffmpeg). Eagerly built
   log-mel `WhisperDataset` (fixed 30 s window, labels tokenized, -100 padding).
   Deterministic train/val split (`--val-split 0.1`, seed 42).
2. **Baseline WER probe** — the stock base model is scored on val utterances
   *before* training; report saved as `report_baseline.json`.
3. **LoRA** — PEFT `LoraConfig(r=8, alpha=32, dropout=0.05)` targeting
   `q_proj` + `v_proj` (≈ <1% of params trainable).
4. **Train** — AdamW + linear warmup schedule, grad clipping, optional
   `--max-steps`; logs loss/lr every `--log-every-steps`.
5. **Post-train WER** — fine-tuned model scored; improvement printed
   (baseline → finetuned); `report_finetuned.json` saved.
6. **Export** — merges LoRA into base weights and converts to a **CTranslate2
   int8** model in `<output-dir>/ct2` (with tokenizer/preprocessor copied next
   to `model.bin` so faster-whisper can load it).
7. **Hot swap** — point the platform at the export with zero code changes:
   ```bash
   SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2
   # or stt.model_path in configs/config.yaml
   ```

### Phase 9 — Observability (metrics, logs, health, alerts)

**Prometheus metrics** (`core/metrics.py`) — shared registry; exposed at
`GET /metrics` (Prometheus text format v0.0.4):

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `stt_requests_total` | Counter | status, channel | STT calls |
| `stt_audio_seconds_total` | Counter | channel | Audio processed |
| `stt_latency_seconds` | Histogram | — | STT wall-clock latency |
| `stt_rtf` | Histogram | — | Real-time factor |
| `stt_confidence` | Histogram | — | exp(avg_logprob) proxy |
| `stt_wer` / `stt_cer` / `stt_rtf_mean` | Gauge | eval_set | Eval-run aggregates |
| `tts_requests_total` | Counter | status, channel | TTS calls |
| `tts_chars_total` | Counter | channel | Characters synthesized |
| `tts_latency_seconds` / `tts_rtf` | Histogram | — | TTS latency / RTF |
| `speech_model_loaded` | Gauge | engine | Engine model loaded? |
| `speech_model_load_seconds` | Histogram | — | Model load duration |
| `speech_jobs_total` | Counter | type, status | Batch jobs processed |
| `speech_jobs_active` | Gauge | — | Jobs currently running |
| `speech_queue_depth` | Gauge | — | Pending jobs |
| `http_requests_total` | Counter | method, path, status | HTTP (paths templated to `{id}`) |
| `http_latency_seconds` | Histogram | — | HTTP latency |
| `speech_ws_active` | Gauge | kind | Open WebSockets |
| `speech_errors_total` | Counter | component, type | Errors raised |

**Structured logs** (`core/logging.py`) — one JSON object per line with `ts`,
`level`, `logger`, `service`, **`request_id`**, **`job_id`**, `message` (+ any
`extra` kwargs, + exception traces). Correlation ids flow through
`contextvars`: the middleware sets `request_id` per HTTP request; the worker
sets `job_id` per job. Machine-parseable for SIEM.

**Health** — `GET /health` returns `{status, version, environment,
models: {stt, tts}, queue: {backend, depth}}`. Engines are lazy, so the API
boots and reports healthy even before models load. The Docker image
`HEALTHCHECK` curls this every 30 s (start period 20 s).

**SLO alerts** (`deploy/prometheus/alerts.yml`): `SttWerRegression` (>10% WER),
`SttRtfDegradation` (RTF > 0.5), `SpeechApiDown`, `SpeechJobFailures` (>5/10m),
`SpeechQueueBacklog` (depth > 100 → scale workers), `SpeechModelUnloaded`,
`SttLatencyP95High` (>30 s). Prometheus scrapes `api:8000/metrics` every 15 s.

---

## 6. How the frontend communicates with the backend

### 6.1 What "frontend" means here

The platform ships a **browser demo console** at `/` (single self-contained
HTML page served by the API) plus a headless API service. Any of these is a
"frontend":

- **Demo console** — `http://localhost:8000/` — three tabs: upload an audio
  file for transcription (with segments, PII redaction badges and telemetry),
  stream speech from the microphone over WebSocket (live partial/final
  hypotheses), and synthesize text to speech with an in-page player.
- `scripts/streaming_demo.py` — the canonical terminal client (mic → ASR,
  text → TTS) showing the full wire protocol.
- `speechai` CLI — sync REST/offline operations.
- A custom SPA/mobile app/IVR that calls the REST + WebSocket endpoints below.

The console is served from `speechai/api/ui/index.html` (declared as package
data in `pyproject.toml` so it ships in the Docker image). It is whitelisted
from API-key auth, but every API call it makes still sends `X-API-Key` when
you enter a key in the top-right field.

> ⚠️ **Browser note:** the API does not enable CORS. A browser SPA on a
> different origin must be served behind a same-origin reverse proxy or the API
> must add a CORS middleware. The API is designed for native/server-side
> clients and works perfectly from `curl`, Python, Node, etc.

### 6.2 Transport & conventions

- **Base URL:** `http(s)://<host>:8000` (HTTP) / `ws(s)://<host>:8000` (WS).
- **Auth:** if `api.api_key` is configured, send header `X-API-Key: <key>` on
  every HTTP request except `/health`, `/metrics`, `/docs`, `/redoc`,
  `/openapi.json` (whitelisted). WebSockets pass the key **inside the first
  JSON message** as `api_key` (no headers in the WS handshake). Unauthorized →
  HTTP 401 `{"error": {"code": "unauthorized", ...}}` or WS close code **1008**.
- **Request correlation:** every HTTP response carries an
  `x-request-id: <12-hex>` header, echoed in `TranscribeResponse.request_id`.
- **Errors:** always the envelope `{"error": {"code": "<stable_code>",
  "message": "...", "details"?: ...}}` with the HTTP status of the mapped error
  class (full table in Section 9).
- **Audio encoding:** clients send/receive PCM16 (little-endian, mono) for
  streaming and standard WAV for files. ASR wants 16 kHz; the server resamples
  other rates.

### 6.3 REST sync STT — wire example

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -H "X-API-Key: $KEY" \
  -F "file=@call.wav" -F "language=en" -F "redact=true"
```

Response (`TranscribeResponse`, 200):

```json
{
  "text": "Your account balance is one thousand two hundred dollars.",
  "language": "en",
  "engine": "faster-whisper",
  "segments": [{"text": "Your account balance...", "start": 0.0, "end": 4.2, "confidence": 0.94}],
  "redacted": false,
  "redactions": [],
  "metrics": {
    "latency_seconds": 1.23, "engine_seconds": 1.10, "rtf": 0.29,
    "audio_duration_seconds": 4.2, "confidence": 0.94
  },
  "request_id": "a1b2c3d4e5f6"
}
```

### 6.4 REST sync TTS — wire example

```bash
# single blob (full WAV in one response)
curl -X POST http://localhost:8000/v1/synthesize \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text": "Your balance is $1,250.50.", "speed": 1.0}' -o speech.wav

# low-latency streaming (one WAV chunk per sentence)
curl -N -X POST "http://localhost:8000/v1/synthesize?stream=true" \
  -H "Content-Type: application/json" -d '{"text": "Hello. Goodbye."}'
```

Request body is `SynthesizeRequest {text (1–10000 chars), voice?, speed (0.5–2.0)}`.
Response: `audio/wav` with `Content-Disposition: attachment;
filename="speech_<ts>.wav"`. Validation errors: an empty `text` is rejected by
Pydantic with **422**; a whitespace-only `text` passes schema validation but is
rejected by the pipeline with **400 `validation_error`**.

### 6.5 REST async batch — the full pattern

The pattern a frontend follows for long jobs: **submit → poll → fetch artifact**.

```bash
# 1. Submit (202 Accepted)
curl -X POST http://localhost:8000/v1/jobs/transcribe \
  -H "X-API-Key: $KEY" -F "file=@call.wav" -F "redact=true"
# -> {"job_id": "9f8e7d6c5b4a3210", "status": "queued", "url": "/v1/jobs/9f8e7d6c5b4a3210"}

# 2. Poll until terminal
curl -s http://localhost:8000/v1/jobs/9f8e7d6c5b4a3210 | jq .status
# queued -> running -> succeeded | failed
# succeeded body carries result (transcript) and, for synthesis, audio_url

# 3. Synthesis artifact
curl -o speech.wav http://localhost:8000/v1/jobs/9f8e7d6c5b4a3210/audio

# 4. Cleanup (204)
curl -X DELETE http://localhost:8000/v1/jobs/9f8e7d6c5b4a3210
```

### 6.6 WebSocket ASR — wire protocol

```
1. open  ws://host:8000/v1/ws/transcribe
2. send  {"sample_rate": 16000, "language": "en"}          (JSON text frame)
3. loop  send  <binary frame: raw PCM16 LE mono chunks>     (any size)
         recv  {"type": "partial", "text": "...", ...}      (every ~2.5s while speaking)
         recv  {"type": "final",   "text": "...", ...}      (per completed utterance)
4. send  {"action": "stop"}   (or disconnect) → server flushes last final and closes
```

Minimal Python client (the pattern behind `scripts/streaming_demo.py`):

```python
import json, websockets

async with websockets.connect("ws://localhost:8000/v1/ws/transcribe") as ws:
    await ws.send(json.dumps({"sample_rate": 16000, "language": "en"}))
    for chunk in pcm16_chunks(audio):          # 16 kHz mono int16 bytes
        await ws.send(chunk)
        # ... concurrently read: {"type": "partial"|"final", ...}
    await ws.send(json.dumps({"action": "stop"}))
```

### 6.7 WebSocket TTS — wire protocol

```
1. open  ws://host:8000/v1/ws/synthesize
2. send  {"text": "Your balance is one thousand dollars. Thank you.", "speed": 1.0}
3. recv  <binary frame: WAV blob for sentence 1>    -> start playback now
   recv  <binary frame: WAV blob for sentence 2>
   recv  {"type": "done", "chunks": 2}              (JSON text frame) -> server closes
```

### 6.8 Formal WebSocket protocol spec

The wire protocol is fully specified — message JSON Schemas, binary framing,
state machine, close codes and worked examples — in
[`docs/ws-protocol.md`](docs/ws-protocol.md).

The two WebSocket endpoints are also injected into the **OpenAPI document**
(`GET /openapi.json`, visible in Swagger UI at `/docs`) by
`speechai.api.ws_openapi` (FastAPI does not include WebSocket routes in its
generated schema by default). Each path carries the vendor extension
`x-websocket: true` plus a pointer to the spec above, so tooling can discover
both streaming contracts next to the REST API.

### 6.9 Endpoint reference summary

| Endpoint | Method | Content-Type in | Content-Type out | Purpose |
|---|---|---|---|---|
| `/` | GET | — | `text/html` | Browser demo console (transcribe upload · live mic · TTS player) |
| `/health` | GET | — | JSON | Liveness/readiness incl. models + queue |
| `/metrics` | GET | — | Prometheus text | Scrape endpoint |
| `/v1/models` | GET | — | JSON | Engine/model config |
| `/v1/transcribe` | POST | multipart (`file`, `language`, `redact`) | JSON | Sync STT |
| `/v1/synthesize` | POST | JSON | `audio/wav` | Sync TTS (add `?stream=true` for chunks) |
| `/v1/jobs/transcribe` | POST | multipart | JSON (202) | Async STT |
| `/v1/jobs/synthesize` | POST | JSON | JSON (202) | Async TTS |
| `/v1/jobs/{id}` | GET | — | JSON | Job status/result |
| `/v1/jobs/{id}` | DELETE | — | 204 | Cancel/remove job |
| `/v1/jobs/{id}/audio` | GET | — | `audio/wav` | Download TTS artifact |
| `/v1/ws/transcribe` | WS | binary PCM16 + JSON ctrl | JSON events | Live ASR |
| `/v1/ws/synthesize` | WS | JSON | binary WAV + JSON done | Live TTS |
| `/docs` | GET | — | HTML | Swagger UI (OpenAPI) |

---

## 7. Starting & stopping the servers

Two supported topologies:

- **Without Docker** — bare metal / local dev. You run the **API** and (for
  batch jobs) the **worker** yourself, plus optionally a **Redis** server.
- **With Docker** — `docker compose` runs **API + worker + Redis + Prometheus**
  as one stack.

### 7.1 Without Docker (bare-metal / local dev)

#### 7.1.1 Prerequisites

- Python **≥ 3.10** (`python --version`)
- Git
- (Only if you want real-time WebSocket demos from the repo: `pip install
  sounddevice websockets`)
- (Only for async batch jobs across processes: a Redis server, or Docker to run
  one)

#### 7.1.2 First-time setup

```bash
# 1. Create + activate a virtualenv
python -m venv .venv
source .venv/Scripts/activate          # Git Bash / WSL on Windows
# .venv\Scripts\activate               # cmd / PowerShell

# 2. Install the package (core + dev + inference engines)
pip install -e ".[dev,engines]"

# 3. Download the Piper voice (Whisper auto-downloads on first use)
python scripts/download_models.py

# 4. (Optional) generate demo samples + eval manifest
python scripts/make_sample_audio.py

# 5. (Optional) verify the CLI works end to end
speechai transcribe data/samples/sample_01_account_balance.wav --json
```

#### 7.1.3 Start the API server

```bash
uvicorn speechai.api.app:app --host 0.0.0.0 --port 8000
# or, with auto-reload for development:
make run-api          # = uvicorn ... --reload
```

- Swagger UI: <http://localhost:8000/docs>
- Health: `curl http://localhost:8000/health`
- On first STT/TTS request the model downloads/loads (adds seconds to that one
  request only).
- **Optional env overrides** (set before launching):

  ```bash
  export SPEECHAI_STT__MODEL_SIZE=small
  export SPEECHAI_API__API_KEY=change-me
  export SPEECHAI_QUEUE__BACKEND=redis
  export SPEECHAI_QUEUE__REDIS_URL=redis://localhost:6379/0
  ```

#### 7.1.4 Start the batch worker (separate terminal)

Needed only for the async `/v1/jobs/*` endpoints.

```bash
python -m speechai.workers.batch_worker
# or: make run-worker
```

> ⚠️ **Critical:** with the default `memory` queue backend, the API and the
> worker must run **in the same process** — a separate worker process will never
> see jobs. For a separate worker process, run Redis and set
> `SPEECHAI_QUEUE__BACKEND=redis` (and the redis URL) **in both** terminals:
>
> ```bash
> # once, if you have Docker available:
> docker run -d --name speechai-redis -p 6379:6379 redis:7-alpine
> ```

#### 7.1.5 Verify the running stack

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0","environment":"development",
#  "models":{"stt":false,"tts":false},"queue":{"backend":"memory","depth":0}}

# exercise it:
curl -X POST http://localhost:8000/v1/synthesize \
  -H "Content-Type: application/json" -d '{"text": "Hello from the bank."}' -o out.wav
```

#### 7.1.6 Stop the servers

| Server | Graceful stop | Force stop (if Ctrl+C fails) |
|---|---|---|
| API (`uvicorn`) | **Ctrl+C** in its terminal (lifespan runs `queue.close()` on shutdown) | Git Bash: `netstat -ano \| grep :8000` → `taskkill //PID <pid> //F`; or PowerShell: `Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess \| Stop-Process` |
| Worker | **Ctrl+C** (catches `KeyboardInterrupt`, closes the queue) | same approach with the port/process of the worker |
| Redis (docker) | `docker stop speechai-redis` | `docker rm -f speechai-redis` |

After stopping, confirm nothing listens on the port:
`curl http://localhost:8000/health` should fail to connect.

#### 7.1.7 Restart

- Config change (`configs/config.yaml` or env): **restart the affected
  process** (API and/or worker). Nothing hot-reloads except `--reload` mode,
  which watches code files only.
- Model change (new voice / STT size / fine-tuned path): restart the API.

### 7.2 With Docker (production-like stack)

`docker-compose.yml` defines **four services**:

| Service | Image | Role |
|---|---|---|
| `api` | built from `Dockerfile` (`python:3.12-slim` + `pip install ".[engines]"`) | FastAPI on :8000; env `SPEECHAI_CONFIG=/app/configs/config.yaml`, queue=redis, data dir `/data`; mounts `./data:/data` and `./configs:/app/configs:ro`; healthcheck curls `/health` every 30s |
| `worker` | same image, command `python -m speechai.workers.batch_worker` | Batch worker on the same Redis queue |
| `redis` | `redis:7-alpine` (`--appendonly yes`) | Job queue; healthcheck `redis-cli ping` |
| `prometheus` | `prom/prometheus:latest` | Scrapes `api:8000`; UI on :9090; config mounted from `deploy/prometheus/` |

Named volumes: `redis-data`, `prometheus-data` (survive `docker compose down`).
The host `./data` bind mount means **models, uploads and results live on the
host** and persist across container rebuilds.

#### 7.2.1 Prerequisites & model prep

1. Docker Engine + Compose v2 (`docker compose version`).
2. **Download the Piper voice on the host first** — the container sees it via
   the `./data` mount:

   ```bash
   python scripts/download_models.py
   ```

   > Whisper note: faster-whisper caches HF downloads inside the container's
   > `~/.cache` (ephemeral). For persistent Whisper weights across rebuilds, put
   > a converted CTranslate2 model under `data/models/` and set
   > `SPEECHAI_STT__MODEL_PATH=/data/models/<name>` (e.g. the fine-tune export),
   > or mount an HF cache volume.

#### 7.2.2 Start the stack

```bash
docker compose up --build
# or detached: docker compose up -d --build
```

What happens: image build → api/worker/redis/prometheus start → redis becomes
healthy → api/worker proceed.

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics | head
# Prometheus UI:   http://localhost:9090
# Swagger UI:      http://localhost:8000/docs
```

Optional env pass-through (e.g. enable auth):

```bash
SPEECHAI_API__API_KEY=my-secret docker compose up --build
# compose forwards it via ${SPEECHAI_API__API_KEY:-}
```

#### 7.2.3 Scale workers horizontally

```bash
docker compose up --scale worker=3 -d
docker compose ps          # 3 worker replicas, all polling the same Redis queue
```

#### 7.2.4 Logs & shell

```bash
docker compose logs -f api            # follow API logs
docker compose logs -f worker         # follow worker logs
docker compose logs prometheus
docker compose exec api bash          # shell inside the API container
docker compose exec redis redis-cli llen speech:queue   # inspect queue depth
```

#### 7.2.5 Stop the stack

| Command | Effect |
|---|---|
| `Ctrl+C` (foreground `up`) | Sends SIGINT to the stack → graceful shutdown (compose stops services in dependency order) |
| `docker compose stop` | Stops containers **without removing them**; start again with `docker compose start` (data intact) |
| `docker compose down` | Stops and **removes** containers + default network; **named volumes (`redis-data`, `prometheus-data`) and the host `./data` mount are kept** |
| `docker compose down -v` | Also deletes named volumes (`redis-data`, `prometheus-data`) — full data wipe. Host `./data` is unaffected (it's a bind mount) |
| `docker compose restart` | Restarts services without rebuilding |
| `docker compose up --build -d` | Rebuild after changing `Dockerfile` / `pyproject.toml` / source |

Per-container control: `docker compose stop api worker`,
`docker compose restart prometheus`, `docker rm -f $(docker compose ps -q)`.

#### 7.2.6 What persists where

| Data | Location | Survives `down`? | Survives `down -v`? |
|---|---|---|---|
| Piper voices, uploads, results, samples | host `./data` (bind mount) | ✅ | ✅ |
| Redis AOF | `redis-data` volume | ✅ | ❌ |
| Prometheus TSDB | `prometheus-data` volume | ✅ | ❌ |

### 7.3 Docker vs. bare-metal — quick decision table

| Need | Use |
|---|---|
| Quick dev loop, no Docker | bare-metal + `make run-api` (memory queue, sync endpoints) |
| Async batch jobs in dev | bare-metal + local Redis (`SPEECHAI_QUEUE__BACKEND=redis` on API *and* worker) |
| Production-like stack, dashboards, alerts | `docker compose up --build` |
| More throughput on batch | `docker compose up --scale worker=N -d` |

---

## 8. Troubleshooting & operational notes

| Symptom | Cause / fix |
|---|---|
| `Piper voice model not found at ...` | Run `python scripts/download_models.py` (or `--piper-voice <voice>`); check `tts.voice` matches a file in `data/models/voices/` |
| `Could not load Whisper model` | Check `stt.model_size`/`stt.model_path`; first load downloads from HF (needs network); `pip install -e ".[engines]"` if the module is missing |
| Jobs stay `queued` forever | Worker not running, or **memory queue split across processes** (Section 7.1.4). Run the worker with the same queue backend/URL |
| HTTP 401 on every endpoint | API key enabled; send `X-API-Key: <value>` (whitelisted: `/health`, `/metrics`, `/docs`) |
| WebSocket closes with 1008 | Auth failed — pass `api_key` in the first JSON message |
| WebSocket closes with 1011 | Server-side error — check API logs (they carry `request_id`) |
| HTTP 413 | Upload > `api.max_upload_mb` (default 50 MB) |
| HTTP 503 `model_not_found` / `engine_unavailable` | Model missing or `faster-whisper`/`piper-tts` not installed |
| Slow first request | Model download/load is lazy; subsequent requests are fast. Consider `scripts/download_models.py --whisper-size <size>` to warm |
| Whisper re-downloads in Docker | Compose sets `HF_HOME=/data/hf-cache` (the mounted `./data`), so the cache survives rebuilds — it only re-downloads if you delete `data/hf-cache` |
| `webrtcvad` missing | `vad.backend=energy` works without it; or install engines extra |
| Port 8000 already in use | Change `api.port` / `--port`, or find & kill the stale process (Section 7.1.6) |
| Queue backlog alert | Scale workers: `docker compose up --scale worker=N -d` |
| Windows CPU fine-tuning | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

**Operational checklist** (fuller map in `docs/production-checklist.md`):
set `service.environment=production`, enable `api.api_key`, use the Redis queue
with ≥ 2 worker replicas, keep Prometheus + alerts wired, retain uploads under
`data/uploads`, and run `speechai evaluate data/manifest.jsonl --gate` in CI.

---

## 9. Reference: endpoints, schemas, error codes, env vars

### 9.1 Error codes → HTTP status

| Error code | HTTP | Raised when | Retryable |
|---|---|---|---|
| `validation_error` | 400 | Empty upload, empty synthesis text | no |
| `audio_format_error` | 400 | Undecodable/empty audio | no |
| `unauthorized` | 401 | Missing/wrong `X-API-Key` (or WS key) | no |
| `job_not_found` | 404 | Unknown/expired job id | no |
| `payload_too_large` | 413 | Upload > `max_upload_mb` | no |
| `quota_exceeded` | 429 | (reserved) | no |
| `engine_unavailable` / `model_not_found` | 503 | Engine/model not loadable | yes |
| `transcription_failed` / `synthesis_failed` | 500 | Inference failure | yes |
| `internal_error` | 500 | Unhandled exception | yes |

Response shape: `{"error": {"code": "...", "message": "...", "details"?: ...}}`.

> **Retryable note:** the `retryable` flag is declared on the error classes but
> **no retry logic is implemented yet** — `BatchPipeline.run_job` marks a failed
> job `failed` and moves on. Treat the column as intent, not behavior.

### 9.2 Key Pydantic schemas (`api/schemas.py`)

- `TranscribeResponse`: `text`, `language`, `engine`, `segments[{text,start,end,confidence}]`, `redacted`, `redactions[{type,masked}]`, `metrics{latency_seconds, engine_seconds, rtf, audio_duration_seconds, confidence}`, `request_id`.
- `SynthesizeRequest`: `text` (1–10000), `voice?`, `speed` (0.5–2.0, default 1.0).
- `JobSubmitResponse`: `job_id`, `status`, `url`.
- `JobStatusResponse`: `id`, `type`, `status`, `created_at`, `started_at`, `finished_at`, `error?`, `metrics?`, `result?`, `audio_url?`.
- `HealthResponse`: `status`, `version`, `environment`, `models{stt,tts}`, `queue{backend, depth}`.
- `ModelInfo`: `engine`, `loaded`, `config`.

### 9.3 Environment variables (SPEECHAI_*)

Pattern: `SPEECHAI_<SECTION>__<FIELD>=<value>` — every key in Section 4.2 is
overridable (e.g. `SPEECHAI_STT__MODEL_SIZE`, `SPEECHAI_TTS__VOICE`,
`SPEECHAI_VAD__BACKEND`, `SPEECHAI_REDACTION__MODE`,
`SPEECHAI_EVAL__DEFAULT_WER_TOLERANCE`, `SPEECHAI_API__API_KEY`). Plus
`SPEECHAI_CONFIG` selects the YAML file. Docker compose additionally injects
`SPEECHAI_QUEUE__BACKEND=redis`, `SPEECHAI_QUEUE__REDIS_URL`,
`SPEECHAI_STORAGE__DATA_DIR=/data`, and `SPEECHAI_API__API_KEY` (passthrough).

### 9.4 Useful Makefile targets

| Target | Command |
|---|---|
| `make install` / `make install-engines` | pip install `.[dev]` / `.[dev,engines]` |
| `make run-api` | uvicorn with `--reload` |
| `make run-worker` | batch worker |
| `make evaluate` | `speechai evaluate data/manifest.jsonl` |
| `make demo` | generate samples + transcribe the first one |
| `make docker-up` / `make docker-down` | `docker compose up --build` / `docker compose down` |
| `make test` | `pytest` |
| `make lint` / `make lint-fix` | `ruff check` / `ruff check --fix` |
