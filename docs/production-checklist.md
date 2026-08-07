# Production-readiness checklist (mapped to the role requirements)

This document maps each key requirement to what the platform already delivers and
what remains for a fully hardened production launch. Use it to frame interviews and
the rollout plan.

## 1. Building & deploying STT/ASR and/or TTS solutions

**Delivered**

- [x] Two on-prem engines behind stable protocols: `faster-whisper` (ASR) and
      `Piper` (TTS), installed by default in the Docker image.
- [x] Containerized deployment: `Dockerfile` + `docker-compose` (API, worker,
      Redis, Prometheus) with healthchecks.
- [x] Model lifecycle: download script, warm cache, lazy loading, load-time metrics.
- [x] CPU-first design (CTranslate2 int8, ONNX) — deployable on commodity VMs;
      CUDA via `stt.device=cuda`.

**Remaining**

- [ ] Managed model registry / version pinning (e.g. S3 + hash manifest).
- [ ] Blue-green model rollout (shadow traffic, canary).
- [ ] Horizontal autoscaling (K8s HPA on queue depth) — compose already scales
      workers manually (`--scale worker=3`).

## 2. Real-time / near real-time speech + batch pipelines

**Delivered**

- [x] WebSocket ASR: VAD-gated utterance segmentation with partial + final events.
- [x] WebSocket TTS: sentence-chunked synthesis for low time-to-first-audio.
- [x] Async batch pipeline with job state machine, Redis queue, N workers, TTL
      retention, artifact retrieval.
- [x] Latency SLOs tracked: end-to-end latency, RTF, p95 histograms.

**Remaining**

- [ ] True word-level streaming (e.g. fine-tuned streaming ASR) if interactivity
      requires sub-second partial latency beyond utterance cadence.
- [ ] Backpressure / rate limiting per tenant; per-call latency budgets.

## 3. Python + PyTorch / HuggingFace ecosystem

**Delivered**

- [x] Fast, typed Python 3.10+ codebase (pydantic v2 models everywhere).
- [x] HuggingFace Hub model management for Whisper; `num2words`/`jiwer`/`numpy`/
      `soundfile` stack; onnxruntime/CTranslate2 runtimes.
- [x] Test suite (pytest, pytest-asyncio) with deterministic stub engines.
- [x] **LoRA fine-tuning harness** (`speechai-finetune`): manifest-driven training,
      baseline vs fine-tuned WER reports, adapter merge + CTranslate2 export,
      and a one-line engine swap via `stt.model_path` (see `docs/finetuning.md`
      and the step-by-step `HOW_TO_FINETUNE_FOR_BANKING.md`).
- [x] **MLflow experiment tracking** for `speechai evaluate` / `speechai-finetune`
      (optional extra; best-effort no-op without a URI — see
      `HOW_TO_FINETUNE_FOR_BANKING.md` §5.6).

## 4. Speech pipeline: data prep → training → inference → evaluation → optimization

**Delivered**

- [x] Data prep: sample generation, JSONL/CSV/directory manifests, 16 kHz
      normalization, VAD for segmentation.
- [x] Training/fine-tuning: LoRA on Whisper with before/after WER gates and
      CTranslate2 export (`speechai-finetune`, `docs/finetuning.md`).
- [x] Inference: sync, batch, and streaming paths with engine abstraction.
- [x] Evaluation: WER, CER, RTF, latency, per-utterance + aggregate reports,
      CI regression gates.
- [x] Optimization: int8/float16 quantization, beam-size tuning, language pinning,
      utterance caps, trailing-silence trimming.

**Remaining**

- [ ] GPU/TPU evaluation baselines and cost models.
- [ ] Voice-cloning fine-tuning for TTS.

## 5. Quality metrics: WER, CER, MOS, latency, reliability

**Delivered**

- [x] **WER / CER** per utterance + aggregates (mean/median/p90), jiwer-backed.
- [x] **Latency**: end-to-end, engine time, RTF; Prometheus histograms.
- [x] **Reliability**: error taxonomy, retryable flags, health checks, queue-depth
      and job-failure alerts, structured audit logs.
- [x] **MOS**: architecture is ready for a MOS column/plugin; Piper's published
      naturalness scores are a baseline proxy. Recommend adding **NISQA** for
      automated MOS on synthesized audio.

**Remaining**

- [ ] NISQA plugin + human listening-panel workflow.
- [ ] Production SLI dashboards (Grafana) with burn-rate SLOs.

## 6. Banking-grade hardening

**Delivered**

- [x] PII redaction on ASR output and TTS input (Luhn-validated cards, accounts,
      IFSC, Aadhaar, PAN, SSN, phones, emails).
- [x] On-prem inference (data sovereignty), optional API-key auth, upload caps,
      correlation-id audit logs, retained uploads.

**Remaining**

- [ ] OIDC/SSO, RBAC, tenant isolation, full audit export (SIEM).
- [ ] Encryption at rest for stored audio; KMS-backed keys.
- [ ] Pen-testing, ISO 27001 / PCI-DSS scoping for the speech pipeline.
- [ ] Data retention policy automation (the TTL mechanics exist).
