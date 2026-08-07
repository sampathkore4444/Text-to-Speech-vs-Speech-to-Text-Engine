# Bank Speech AI — Quickstart Cheat Sheet

> The fast reference: essential start/stop commands and `curl` examples.
> For the full walkthrough (every phase, wire protocols, config keys), see
> **`END-TO-END-FLOW.md`**.

---

## 1. First-time setup (once)

```bash
# Python >= 3.10
python -m venv .venv
source .venv/Scripts/activate          # Git Bash / WSL on Windows
# .venv\Scripts\activate               # cmd / PowerShell
pip install -e ".[dev,engines]"

python scripts/download_models.py                    # Piper voice (required)
python scripts/download_models.py --whisper-size base  # optional: warm Whisper cache

python scripts/make_sample_audio.py                  # optional: demo samples + manifest
speechai models                                      # sanity check
```

---

## 2. Start & stop — WITHOUT Docker

```bash
# API server (Swagger UI at http://localhost:8000/docs)
uvicorn speechai.api.app:app --host 0.0.0.0 --port 8000
# or dev mode with auto-reload:  make run-api

# Batch worker (2nd terminal) — only needed for async /v1/jobs/* endpoints
python -m speechai.workers.batch_worker
# or: make run-worker

# Local Redis (only if you run API + worker as SEPARATE processes)
docker run -d --name speechai-redis -p 6379:6379 redis:7-alpine
export SPEECHAI_QUEUE__BACKEND=redis                 # set in BOTH terminals
export SPEECHAI_QUEUE__REDIS_URL=redis://localhost:6379/0
```

**Verify:** `curl http://localhost:8000/health` · demo console UI: http://localhost:8000/

**Stop:**

| What | Graceful | Force (if Ctrl+C fails) |
|---|---|---|
| API | `Ctrl+C` in its terminal | `netstat -ano \| grep :8000` → `taskkill //PID <pid> //F` |
| Worker | `Ctrl+C` | same, using the worker's process |
| Redis | `docker stop speechai-redis` | `docker rm -f speechai-redis` |

> ⚠️ With the default `memory` queue, a **separate** worker process never sees
> jobs — the queue is per-process. Use Redis (above) or run worker in-process.

---

## 3. Start & stop — WITH Docker (full stack: API + worker + Redis + Prometheus)

```bash
# Start (build + up, foreground or detached)
docker compose up --build
docker compose up -d --build

# Verify
curl http://localhost:8000/health
# Demo console UI: http://localhost:8000/  (transcribe upload · live mic · TTS player)
# Prometheus UI: http://localhost:9090

# Scale workers horizontally
docker compose up --scale worker=3 -d

# Logs / shell
docker compose logs -f api worker
docker compose exec api bash

# Stop
docker compose down          # keeps volumes (redis-data, prometheus-data) + ./data
docker compose down -v       # also wipes named volumes (full data reset)
docker compose stop          # stop without removing; start again with: docker compose start
docker compose restart       # restart without rebuild
```

> **Before first `docker compose up`:** run `python scripts/download_models.py`
> on the host — the container sees models via the `./data` bind mount. In
> Docker the Whisper cache is kept in `./data/hf-cache` (`HF_HOME=/data/hf-cache`),
> so it survives rebuilds too.

---

## 4. REST API — curl examples

```bash
# Set once (only if auth enabled: SPEECHAI_API__API_KEY=...)
KEY="your-api-key"

# --- System ---------------------------------------------------------------
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/v1/models

# --- Sync STT (multipart) -------------------------------------------------
curl -X POST http://localhost:8000/v1/transcribe \
  -H "X-API-Key: $KEY" \
  -F "file=@call.wav" -F "language=en" -F "redact=true"

# --- Sync TTS (JSON -> WAV) ------------------------------------------------
curl -X POST http://localhost:8000/v1/synthesize \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text": "Your balance is $1,250.50.", "speed": 1.0}' -o speech.wav

# low-latency: one WAV chunk per sentence
curl -N -X POST "http://localhost:8000/v1/synthesize?stream=true" \
  -H "Content-Type: application/json" -d '{"text": "Hello. Goodbye."}'

# --- Async batch: submit -> poll -> artifact -> delete ----------------------
curl -X POST http://localhost:8000/v1/jobs/transcribe \
  -H "X-API-Key: $KEY" -F "file=@call.wav" -F "redact=true"
# -> {"job_id": "9f8e7d6c5b4a3210", "status": "queued", "url": "/v1/jobs/9f8e7d6c5b4a3210"}

curl -s http://localhost:8000/v1/jobs/9f8e7d6c5b4a3210 | jq .status
# queued -> running -> succeeded | failed

curl -o speech.wav http://localhost:8000/v1/jobs/9f8e7d6c5b4a3210/audio   # synthesis only
curl -X DELETE http://localhost:8000/v1/jobs/9f8e7d6c5b4a3210              # 204
```

---

## 5. WebSocket — streaming examples

Extra deps: `pip install websockets sounddevice`

```bash
# Transcribe from mic (partial/final events print live)
python scripts/streaming_demo.py --transcribe

# Transcribe a 16 kHz mono WAV instead
python scripts/streaming_demo.py --transcribe --file call.wav

# Synthesize text and play it back sentence-by-sentence
python scripts/streaming_demo.py --synthesize "Your balance is one thousand dollars." --speed 1.1

# Save the audio instead of playing
python scripts/streaming_demo.py --synthesize "Hello from the bank." --output out.wav
```

**ASR protocol** (`/v1/ws/transcribe`): send JSON config `{"sample_rate": 16000,
"language": "en", "api_key": ...}` → stream raw **PCM16 LE mono** chunks →
receive `{"type": "partial"|"final", "text": ..., ...}` → send `{"action":
"stop"}`.

**TTS protocol** (`/v1/ws/synthesize`): send `{"text": "...", "speed": 1.0,
"api_key": ...}` → receive one **WAV blob per sentence** → final
`{"type": "done", "chunks": N}`.

**Formal spec:** `docs/ws-protocol.md` — full JSON Schemas, framing, close
codes, examples. Both WS endpoints also appear in Swagger UI (`/docs`) via the
injected `x-websocket` OpenAPI extension (`GET /openapi.json`).

> Server failures arrive as a JSON `error` event (e.g. `model_not_found`)
> before the socket closes with `1011` — the demo console shows them as a
> banner instead of hanging silently.

Minimal Python client:

```python
import json, websockets

async with websockets.connect("ws://localhost:8000/v1/ws/transcribe") as ws:
    await ws.send(json.dumps({"sample_rate": 16000, "language": "en"}))
    for chunk in pcm16_chunks(audio):          # 16 kHz mono int16 bytes
        await ws.send(chunk)
    await ws.send(json.dumps({"action": "stop"}))
```

---

## 6. CLI — offline operations

```bash
speechai transcribe data/samples/sample_01_account_balance.wav --json
speechai synthesize "Your balance is one thousand dollars." -o out.wav --speed 1.0
speechai evaluate data/manifest.jsonl --gate        # WER/CER/RTF report + regression gates
speechai models

# Fine-tune Whisper with LoRA (needs: pip install -e ".[finetune]")
speechai-finetune --data data/manifest.jsonl --base-model openai/whisper-base \
    --output-dir data/models/finetuned --language en
# swap the fine-tuned model in with zero code changes:
export SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2
```

---

## 7. Useful env vars

| Variable | Effect |
|---|---|
| `SPEECHAI_CONFIG=/path/config.yaml` | Which YAML to load (default `configs/config.yaml`) |
| `SPEECHAI_API__API_KEY=secret` | Enable `X-API-Key` auth (all endpoints except `/health`, `/metrics`, `/docs`) |
| `SPEECHAI_API__PORT=9000` | Change API port |
| `SPEECHAI_STT__MODEL_SIZE=small` | STT model (tiny→large-v3) |
| `SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2` | Use a local CTranslate2 model |
| `SPEECHAI_TTS__VOICE=en_US-amy-medium` | Piper voice |
| `SPEECHAI_QUEUE__BACKEND=redis` | Redis queue for multi-process/multi-worker |
| `SPEECHAI_QUEUE__REDIS_URL=redis://localhost:6379/0` | Redis URL |
| `SPEECHAI_STORAGE__DATA_DIR=/data` | Data root (Docker sets this) |
| `SPEECHAI_REDACTION__MODE=redact` | PII mode: `mask` \| `redact` \| `none` |

---

## 8. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `Piper voice model not found` | Run `python scripts/download_models.py` |
| Jobs stuck `queued` | Worker not running, or memory queue split across processes → use Redis (Section 2) |
| HTTP 401 everywhere | Add `-H "X-API-Key: $KEY"` |
| WebSocket closes `1008` | Pass `api_key` inside the first JSON message |
| HTTP 413 | Upload > 50 MB (`api.max_upload_mb`) |
| Slow first request | Model downloads/loads lazily — warm with `download_models.py --whisper-size base` |

---

## 9. Makefile one-liners

```bash
make install          # pip install -e ".[dev]"
make install-engines  # pip install -e ".[dev,engines]"
make run-api          # uvicorn ... --reload
make run-worker       # python -m speechai.workers.batch_worker
make test             # pytest
make lint             # ruff check
make docker-up        # docker compose up --build
make docker-down      # docker compose down
```

---

See **`END-TO-END-FLOW.md`** for architecture, every phase in detail, wire
protocols, and the full start/stop runbook.
