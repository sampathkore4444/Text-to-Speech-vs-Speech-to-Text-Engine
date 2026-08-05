# Bank Speech AI Platform

A **production-grade speech AI platform** (STT + TTS) built for banking: on-premise
ASR with VAD, bank-specific PII redaction, real-time WebSocket streaming, an async
batch pipeline, an evaluation harness (WER / CER / RTF / latency) and full
observability. Everything runs locally on CPU with open-source models, so customer
audio never leaves the bank's infrastructure.

> **Status:** v0.1.0 — clean architecture, tested core, containerized deployment.

---

## Highlights

| Requirement (job description)            | Implementation |
|------------------------------------------|----------------|
| Build & deploy STT/ASR and/or TTS        | faster-whisper (CTranslate2) ASR + Piper TTS, both on-prem |
| Real-time / near real-time speech        | WebSocket streaming: VAD-gated utterance ASR + sentence-chunked TTS |
| Python + PyTorch / HuggingFace ecosystem | PyTorch-free fast inference via CTranslate2/ONNX; HF Hub model management |
| Full speech pipeline knowledge           | Data prep (`make_sample_audio`, manifest loader) → fine-tune hooks → inference → evaluation → optimization (int8/float16, beam size) |
| Quality metrics: WER, CER, MOS, latency, reliability | Evaluation harness (`speechai evaluate`) + Prometheus SLIs + SLO alerts |
| Banking-grade safety                     | PII redaction (cards w/ Luhn, accounts, IFSC, Aadhaar, PAN, SSN, phones, emails) on both ASR output and TTS input |

---

## Architecture

```
                        ┌────────────────────────────────────────────────┐
                        │                    Clients                     │
                        │  REST (sync + async)   ·  WebSocket streaming   │
                        └───────────────┬────────────────────────────────┘
                                        │
                      ┌─────────────────▼──────────────────┐
                      │         API service (FastAPI)       │
                      │  /v1/transcribe  /v1/synthesize     │
                      │  /v1/jobs/*      /v1/ws/*           │
                      │  Observability middleware           │
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

**Layers:** `speechai.core` (config/logging/metrics/errors) · `speechai.audio`
(IO/VAD) · `speechai.stt` / `speechai.tts` (engine-abstracted) ·
`speechai.redaction` (PII) · `speechai.pipeline` (jobs/queue) · `speechai.eval` ·
`speechai.api` (REST+WS) · `speechai.cli`.

See [docs/architecture.md](docs/architecture.md) for design decisions and
[docs/production-checklist.md](docs/production-checklist.md) for the production-readiness
map.

---

## Quickstart

Requires Python ≥ 3.10. Models download on first use (Whisper) or via script (Piper).

```bash
# 1. Install (core deps for tests; +engines for local inference)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,engines]"

# 2. Fetch the Piper TTS voice (Whisper downloads itself on first load)
python scripts/download_models.py

# 3. Generate bank-domain sample audio (TTS -> WAV -> STT demo loop)
python scripts/make_sample_audio.py

# 4. CLI: transcribe, synthesize, evaluate
speechai transcribe data/samples/sample_01_account_balance.wav --json
speechai synthesize "Your balance is one thousand two hundred dollars." -o out.wav
speechai evaluate data/manifest.jsonl --gate     # WER/CER/RTF report

# 5. Serve the API (swagger at http://localhost:8000/docs)
uvicorn speechai.api.app:app --host 0.0.0.0 --port 8000

# 6. Run the batch worker in a second terminal
python -m speechai.workers.batch_worker
```

### Docker (production-like stack: API + worker + Redis + Prometheus)

```bash
docker compose up --build
curl http://localhost:8000/health
# Prometheus dashboards: http://localhost:9090
```

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness/readiness incl. model + queue state |
| `/metrics` | GET | Prometheus exposition |
| `/v1/transcribe` | POST | Upload audio → JSON transcript (multipart: `file`, `language`, `redact`) |
| `/v1/synthesize` | POST | Text → WAV (`{"text": "...", "speed": 1.0}`, `?stream=true` for low latency) |
| `/v1/jobs/transcribe` | POST | Async transcription job → `202 {job_id}` |
| `/v1/jobs/synthesize` | POST | Async synthesis job → `202 {job_id}` |
| `/v1/jobs/{id}` | GET/DELETE | Job status / cancel |
| `/v1/jobs/{id}/audio` | GET | Download synthesized WAV artifact |
| `/v1/models` | GET | Engine/model config |
| `/v1/ws/transcribe` | WS | Real-time ASR (send PCM16 chunks, receive partial/final JSON) |
| `/v1/ws/synthesize` | WS | Real-time TTS (send text, receive per-sentence WAV) |

Auth: set `SPEECHAI_API__API_KEY` and send `X-API-Key` on everything except
`/health`, `/metrics`, `/docs`.

### Streaming ASR example

```python
import json, websockets

async with websockets.connect("ws://localhost:8000/v1/ws/transcribe") as ws:
    await ws.send(json.dumps({"sample_rate": 16000, "language": "en"}))
    for chunk in pcm16_chunks(audio):          # 16 kHz mono int16
        await ws.send(chunk)                    # raw binary frames
        for event in ...: pass                  # {"type": "partial"|"final", "text": ...}
    await ws.send(json.dumps({"action": "stop"}))
```

---

## Evaluation (WER / CER / RTF / latency)

```bash
speechai evaluate data/manifest.jsonl --report data/eval/report.json --gate
```

- Per-utterance **WER** (word error rate) and **CER** (character error rate) via
  `jiwer`; **RTF** (real-time factor) and end-to-end latency per file.
- Aggregate report (mean / median / p90) + worst-utterance ranking, exported as
  JSON; aggregates are pushed to Prometheus gauges (`stt_wer`, `stt_rtf_mean`, ...)
  for regression alerting.
- `--gate` fails CI when mean WER > 10% or mean RTF > 0.5 (configurable).
- **MOS:** real MOS requires human listening panels or a model like NISQA — the
  harness is structured so a MOS column/plugin can be added; Piper voices have
  published naturalness scores.

Evaluation datasets can be JSONL (`{"audio": ..., "reference": ...}`), CSV, or a
directory of `wav + txt` pairs — drop in a Common Voice / internal call-audio subset.

---

## Security (banking)

- **On-premise inference** — audio never leaves your network.
- **PII redaction** on ASR output *and* TTS input: cards (Luhn-validated),
  account numbers, IFSC, Aadhaar, PAN, SSN, phone numbers, emails.
  `mask` mode keeps the last 4 digits; `redact` removes entirely. TTS input is
  normalized so sensitive values are spoken as "redacted", never as digits.
- Upload size caps, content validation, optional API-key auth, structured audit
  logs with request/job correlation ids, retained uploads for compliance.

---

## Configuration

Single YAML file (`configs/config.yaml`) with `SPEECHAI_*` env overrides
(`SPEECHAI_STT__MODEL_SIZE=small`). See `.env.example` for the full list.

| Key | Default | Notes |
|---|---|---|
| `stt.model_size` | `base` | `tiny`→`large-v3` or HF id; `int8` quantization on CPU |
| `stt.model_path` | `` | local CTranslate2 dir (LoRA fine-tuned export) - takes precedence |
| `stt.language` | auto | pin e.g. `en` for lower latency |
| `tts.voice` | `en_US-lessac-medium` | fetch via `download_models.py` |
| `queue.backend` | `memory` | `redis` for scale-out (compose default) |
| `redaction.mode` | `mask` | `mask`/`redact`/`none` |
| `vad.backend` | `auto` | WebRTC preferred, energy fallback |

---

## Project layout

```
src/speechai/
  core/        config · structured logging · prometheus metrics · errors
  audio/       AudioBuffer · resampling · VAD + streaming segmenter
  stt/         engine protocol · faster-whisper · post-processing · streaming
  tts/         engine protocol · Piper · bank text normalization · streaming
  redaction/   Luhn-validated PII redaction
  pipeline/    job model · memory/Redis queue · batch orchestrator
  eval/        WER/CER/RTF/latency · dataset loading · regression gates
  finetune/    LoRA fine-tuning of Whisper + CTranslate2 export (`speechai-finetune`)
  api/         FastAPI app · REST · WebSockets · middleware · schemas
  workers/     batch worker (scale-out)
  cli/         `speechai` command line
scripts/       model downloads · sample audio generation
deploy/        Prometheus scrape + SLO alerts
tests/         unit + API contract + pipeline tests (stub engines)
```

---

## Fine-tuning

Adapt Whisper to your bank's vocabulary and call audio with LoRA, then swap the
engine to the exported checkpoint with one config change:

```bash
speechai-finetune --data data/manifest.jsonl --base-model openai/whisper-base \
    --output-dir data/models/finetuned --language en
SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2
```

Reports WER before/after and exports a CTranslate2 model — see
[docs/finetuning.md](docs/finetuning.md).

## Roadmap

- [x] Fine-tuning hooks (LoRA for Whisper → [docs/finetuning.md](docs/finetuning.md)); voice cloning for TTS remains
- [ ] Speaker diarization + per-speaker attribution for call analytics
- [ ] MOS plugin (NISQA) and human-eval workflow
- [ ] AuthN/Z (OIDC), rate limiting, audit export (SIEM)
- [ ] Load testing + autoscaling (HPA) manifests
