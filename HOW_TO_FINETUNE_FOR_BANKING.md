# How to Fine-Tune Whisper for Banking (LoRA) — Complete Step-by-Step Guide

> This guide walks you through **every step** of adapting a general-purpose Whisper
> speech-to-text model to a bank's domain using **LoRA** (Low-Rank Adaptation),
> from installing the tooling to swapping the fine-tuned model into the running
> platform and gating it with WER evaluation.
>
> Everything described here is implemented in this repository:
>
> | What | Where |
> |---|---|
> | Training CLI (`speechai-finetune`) | `src/speechai/finetune/train.py` |
> | PyTorch dataset builder | `src/speechai/finetune/dataset.py` |
> | Manifest loader (JSONL / CSV / wav+txt dir) | `src/speechai/eval/loader.py` |
> | Starter sample data generator | `scripts/make_sample_audio.py` |
> | Evaluation harness (WER/CER/RTF + gates) | `src/speechai/eval/*` |
> | faster-whisper engine (loads the fine-tuned model) | `src/speechai/stt/whisper_engine.py` |
> | Configuration (model swap) | `configs/config.yaml` (`stt.model_path`) |
>
> A shorter overview lives in [`docs/finetuning.md`](docs/finetuning.md); this
> document is the deep-dive companion with worked examples.

---

## Table of contents

1. [What you are building — and why LoRA](#1-what-you-are-building--and-why-lora)
2. [The full pipeline at a glance](#2-the-full-pipeline-at-a-glance)
3. [Prerequisites](#3-prerequisites)
4. [Step 0 — Install the fine-tuning stack](#step-0--install-the-fine-tuning-stack)
5. [Step 1 — Understand the data contract (manifests)](#step-1--understand-the-data-contract-manifests)
6. [Step 2 — Prepare your training data](#step-2--prepare-your-training-data)
7. [Step 3 — Validate your data before training](#step-3--validate-your-data-before-training)
8. [Step 4 — Smoke run (prove the machinery works)](#step-4--smoke-run-prove-the-machinery-works)
9. [Step 5 — The real training run](#step-5--the-real-training-run)
10. [Step 6 — Read and interpret the results](#step-6--read-and-interpret-the-results)
11. [Step 7 — Swap the fine-tuned model into the platform](#step-7--swap-the-fine-tuned-model-into-the-platform)
12. [Step 8 — Evaluate and gate the new model](#step-8--evaluate-and-gate-the-new-model)
13. [Step 9 — Production considerations](#step-9--production-considerations)
14. [Troubleshooting](#14-troubleshooting)
15. [Reference — every flag, output artifact, and file](#15-reference--every-flag-output-artifact-and-file)

---

## 1. What you are building — and why LoRA

**The problem.** Whisper is trained on generic Internet audio. Bank audio is
different in predictable ways:

- **Vocabulary:** product names ("fixed deposit", "IFSC", "net banking"), amounts,
  dates, account jargon.
- **Acoustics:** telephony/IVR codecs (8–16 kHz, compressed), headset mics,
  accents, background noise, music-on-hold.
- **Format:** short utterances, numeric-heavy content ("your balance is one
  thousand two hundred fifty dollars and fifty cents").

A stock Whisper model will often transcribe these with domain-specific errors
("fixed deposit" → "fix deposit", "one two five zero" → "1250"). Fine-tuning
teaches the model your vocabulary and acoustic conditions.

**Why LoRA specifically.** LoRA keeps the original model weights **frozen** and
trains only small low-rank adapter matrices injected into the attention layers
(`q_proj`, `v_proj`). Consequences:

- **Tiny footprint:** only ~0.1–1% of parameters are trainable (see the
  "trainable params" line in the output), so training works on a single GPU —
  or even a laptop CPU.
- **Fast:** a few epochs over a couple of hours of audio is usually enough.
- **Safe:** the base model is never modified, so you keep a pristine copy and a
  clean rollback path.
- **Portable:** at the end the adapters are *merged back* into the base weights
  and **converted to CTranslate2** (`model.bin`), which is exactly the format
  the platform's `faster-whisper` engine loads — no code changes needed.

**The banking angle.** On-premise training and inference mean customer audio
never leaves your infrastructure. Everything below runs on your own machines.

---

## 2. The full pipeline at a glance

```
 ┌─────────────┐   ┌──────────────────┐   ┌────────────────────────────┐
 │  Manifest   │ → │  WhisperDataset  │ → │  LoRA training (PyTorch)   │
 │  (audio +   │   │  log-mel + labels│   │  frozen base + q/v adapters│
 │  reference) │   │  ≤30 s windows   │   └────────────┬───────────────┘
 └─────────────┘   └──────────────────┘                │ merge + export
                                                       ▼
                                         ┌──────────────────────────┐
                                         │  CTranslate2 model dir   │
                                         │  model.bin + tokenizer   │
                                         └────────────┬─────────────┘
                                                      │ stt.model_path
                                                      ▼
                              ┌───────────────────────────────────────┐
                              │  Platform API / CLI (faster-whisper)  │
                              │  speechai transcribe / REST / WS     │
                              └───────────────────────────────────────┘
```

The pipeline is:

1. **Prepare data** — audio files + reference transcripts in a *manifest*.
2. **Train** — `speechai-finetune` loads the base Whisper model, attaches LoRA,
   trains for N epochs, and reports WER **before** (baseline) and **after**.
3. **Export** — the script merges the LoRA adapters into the weights and
   converts them to a CTranslate2 `model.bin` (quantized `int8` by default).
4. **Swap** — point `stt.model_path` at the exported directory.
5. **Evaluate & gate** — `speechai evaluate` proves the new model holds the WER
   bar before it goes live.

---

## 3. Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 |
| RAM | 8 GB | 16 GB+ (Whisper `base` training on CPU) |
| Disk | 10 GB free | 30 GB+ (models + audio) |
| GPU | none (CPU works) | 1× CUDA GPU for faster training |
| Network | access to `huggingface.co` **once** (to download the base model) | — |

> ⚠️ **HF Hub access.** `speechai-finetune` downloads the base Whisper model
> from the Hugging Face Hub on first use (`openai/whisper-base` etc.). If your
> bank's training box is air-gapped, pre-download the weights on a connected
> machine and pass `--base-model /path/to/local/model` instead (any local
> directory with the HF Whisper files works).

---

## Step 0 — Install the fine-tuning stack

**0.1 — Create/activate the virtual environment** (if you haven't already):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows Git Bash / WSL:
source .venv/Scripts/activate
```

**0.2 — Install the base package plus the `finetune` extra:**

```bash
pip install -e ".[dev,engines,finetune]"
```

That installs the platform plus the LoRA training stack. The `finetune` extra
adds exactly (`pyproject.toml`):

| Package | Version | Role |
|---|---|---|
| `torch` | `>=2.1` | Deep-learning framework |
| `transformers` | `>=4.44,<5` | Whisper model + processor (**pinned `<5`** — the Whisper forward signature changed in 5.x and the LoRA recipe is tested on 4.x) |
| `peft` | `>=0.10` | LoRA (Parameter-Efficient Fine-Tuning) |
| `accelerate` | `>=0.30` | Optional launcher for multi-GPU training |
| `safetensors` | `>=0.4` | Safe weight serialization |

**0.3 — (Windows only) install CPU-only PyTorch:**

The default `pip install torch` on Windows drags in a multi-GB CUDA build.
For CPU-only training install the smaller build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(Skip this if you have an NVIDIA GPU and want CUDA.)

**0.4 — Verify the CLI is on PATH:**

```bash
speechai-finetune --help        # or: python -m speechai.finetune.train --help
speechai models
```

If `speechai-finetune` is not found, the entry point is registered by the
editable install; on Windows the script lives at `.venv/Scripts/speechai-finetune.exe`.

> ✅ **Checkpoint:** `speechai-finetune --help` prints the full flag list
> (no torch import is needed to print help, so this works even before Step 0.3).

---

## Step 1 — Understand the data contract (manifests)

The training script reads the **same manifest format** as the evaluation
harness (`src/speechai/eval/loader.py`). One manifest = your whole dataset.

### Format A — JSONL (recommended for large/curated sets)

One JSON object per line with `audio` and `reference` keys:

```jsonl
{"audio": "data/calls/call_0001.wav", "reference": "Your account balance is one thousand two hundred fifty dollars and fifty cents as of today."}
{"audio": "data/calls/call_0002.wav", "reference": "Please verify the last four digits of your card to continue."}
{"audio": "data/calls/call_0003.wav", "reference": "We detected unusual activity on your account. Press one to confirm a recent transaction."}
```

### Format B — CSV

Two columns, header `audio,reference`:

```csv
audio,reference
data/calls/call_0001.wav,Your account balance is one thousand two hundred fifty dollars and fifty cents as of today.
data/calls/call_0002.wav,Please verify the last four digits of your card to continue.
```

### Format C — Directory of `wav + txt` pairs

Drop `<name>.wav` and a matching `<name>.txt` (same stem, same folder). The
loader pairs them automatically:

```
data/calls/
  call_0001.wav   + call_0001.txt   ("Your account balance is ...")
  call_0002.wav   + call_0002.txt
  ...
```

Pass the **directory path** as `--data` (the loader detects it).

> **Relative vs absolute paths.** Paths in the manifest are resolved relative
> to your current working directory when you launch training. Absolute paths
> always work. The generated starter manifest (Step 2.1) uses absolute paths.

---

## Step 2 — Prepare your training data

This is the step that determines 80% of your results. Do it carefully.

### 2.1 — (Optional but recommended) Generate starter data with the built-in script

The platform can synthesize **bank-domain sample audio** with its own TTS voice
so you can prove the whole loop works without recording anything:

```bash
python scripts/make_sample_audio.py
```

This writes:

```
data/samples/sample_01_account_balance.wav   + sample_01_account_balance.txt
data/samples/sample_02_card_verification.wav + sample_02_card_verification.txt
data/samples/sample_03_interest_rate.wav     + sample_03_interest_rate.txt
data/samples/sample_04_fraud_alert.wav       + sample_04_fraud_alert.txt
data/samples/sample_05_appointment.wav       + sample_05_appointment.txt
data/manifest.jsonl                          ← 5-utterance manifest
```

The five phrases cover exactly the banking vocabulary you care about: account
balances, card verification, interest rates, fraud alerts, appointments.
**Treat these only as a smoke test** — TTS audio is clean and will not
replicate real call acoustics.

### 2.2 — Collect real audio (for actual gains)

Aim for **1–2 hours minimum of transcribed audio** (more is better — a few
hours of representative calls is a sweet spot for LoRA; huge corpora are fine
too, see Step 9 for scaling).

Recommended collection rules:

| Rule | Why |
|---|---|
| **Use real production audio** (calls, IVR recordings, voicemails) | Domain acoustic transfer comes from *your* mics/codecs, not synthetic audio |
| **16 kHz mono WAV preferred**; MP3/FLAC/OGG also accepted (decoded with `soundfile`) | The dataset resamples everything to 16 kHz internally |
| **Split long recordings into ≤ 30 s chunks** | Whisper trains on a fixed 30 s window; anything longer is **truncated** (`--max-audio-seconds`, default 30.0) |
| **Transcribe verbatim** | The reference should match exactly what is said — including "um", numbers-as-spoken, etc. |
| **Keep references consistent with how you want output** | If you want amounts spoken as words, write them as words in the reference |
| **Remove/redact PII from your corpus** | Train on anonymized data; don't train a model that memorizes card numbers |
| **Balanced coverage** | Make sure all your key phrases/products appear multiple times |

> **Tip — splitting long calls.** Split by silence using any standard tool, e.g.
> `ffmpeg -i call.wav -af silencedetect=noise=-35dB:d=0.4 -f null -` to find
> silence, then `ffmpeg -i call.wav -ss <start> -t <dur> -ar 16000 -ac 1 chunk.wav`.
> Keep each chunk a single coherent utterance — Whisper learns best from
> utterance-length units.

### 2.3 — Write the manifest

Point all three formats at your audio. Example with a mix of JSONL lines:

```bash
# from the repo root
{
  echo '{"audio": "data/calls/call_0001.wav", "reference": "Your account balance is one thousand two hundred fifty dollars and fifty cents as of today."}'
  echo '{"audio": "data/calls/call_0002.wav", "reference": "Please verify the last four digits of your card to continue."}'
} > data/manifest.jsonl
```

Or, if you have a folder of `wav + txt` pairs, just pass the folder:

```bash
speechai-finetune --data data/calls/ ...   # directory form
```

### 2.4 — Understand the dataset internals (so you can size things)

`WhisperDataset` (`src/speechai/finetune/dataset.py`) does this per utterance:

1. **Decode** with the platform's own audio stack (`soundfile`) — no ffmpeg
   needed.
2. **Resample** to 16 kHz mono float32.
3. **Truncate** to `min(max_audio_seconds, 30 s)` — the Whisper hard window
   (`WHISPER_MAX_SAMPLES = 480000` samples).
4. **Log-mel features** — `(80 mel bins × up to 3000 frames)` via the HF
   feature extractor, padded to the fixed 30 s window.
5. **Labels** — tokenized reference, capped at `LABEL_MAX_LENGTH = 448` tokens.

Everything is built **eagerly in memory** (`self._items = [...]`). Budget
roughly 1–3 MB of RAM per second of audio for `base`-size features; tens of
hours of audio need the disk-backed approach in Step 9.

---

## Step 3 — Validate your data before training

Before spending compute, confirm the data is usable and the domain is actually
being transcribed poorly (that's your motivation baseline):

```bash
# 1. Can the platform decode and transcribe a sample already?
speechai transcribe data/samples/sample_01_account_balance.wav --json

# 2. Baseline domain quality on your real data
speechai evaluate data/manifest.jsonl
```

`speechai evaluate` runs the stock model over your manifest and prints mean /
median / p90 **WER, CER, RTF**. High WER on bank vocabulary = good
fine-tuning opportunity. Save this number — the training script computes its
own baseline probe too, but an independent one never hurts.

> **Gotcha:** if `speechai evaluate` reports errors like `FileNotFoundError` for
> every line, your manifest paths are wrong relative to your CWD — fix paths
> before training.

---

## Step 4 — Smoke run (prove the machinery works)

Never start with the big run. Prove the full loop (data load → LoRA attach →
train a few steps → export → swap) on a tiny budget first.

**4.1 — A 3-minute smoke run with the tiny model:**

```bash
speechai-finetune \
  --data data/manifest.jsonl \
  --base-model openai/whisper-tiny \
  --output-dir data/models/smoke \
  --language en \
  --max-steps 5 \
  --batch-size 1 \
  --epochs 1
```

Notes:

- `openai/whisper-tiny` downloads ~40 MB — much faster than `base`.
- `--max-steps 5` stops training after **5 optimizer steps** — the run finishes
  in seconds and validates every stage.
- The baseline WER probe still runs (on `val_split` of your data).

**4.2 — What a healthy smoke run looks like:**

```text
=== Bank Speech AI - Whisper LoRA fine-tuning ===
base model : openai/whisper-tiny
device     : cpu
output dir : data/models/smoke

manifest    : data/manifest.jsonl (5 utterances)
train/val   : 4 / 1

loading processor + base model (first run downloads weights)...

probing baseline WER on 1 val utterances...
BASELINE WER  WER 0.000  CER 0.000  RTF 0.42  (n=1)
trainable params: 786,432 / 37,760,512 (2.083%)

training (1 epochs, batch 1, lr 0.0001, grad_accum 1)...
step 5  loss 0.8123  lr 1.00e-04
epoch 1 complete  mean loss 0.8002
training finished in 12.3s (5 optimizer steps)

evaluating fine-tuned model...
FINE-TUNED WER  WER 0.000  CER 0.000  RTF 0.55  (n=1)

adapter + processor saved to data/models/smoke
CTranslate2 model exported to: data/models/smoke/ct2
swap the platform to the fine-tuned model with:
    SPEECHAI_STT__MODEL_PATH=data/models/smoke/ct2

Done.
```

**4.3 — Verify the smoke model actually loads and transcribes:**

```bash
export SPEECHAI_STT__MODEL_PATH=data/models/smoke/ct2
speechai transcribe data/samples/sample_01_account_balance.wav --json
unset SPEECHAI_STT__MODEL_PATH     # back to normal afterwards
```

If that prints a transcript, the entire pipeline (train → merge → CT2 export →
faster-whisper load) works on your machine. Now do it for real.

---

## Step 5 — The real training run

### 5.1 — The command

```bash
speechai-finetune \
  --data data/manifest.jsonl \
  --base-model openai/whisper-base \
  --output-dir data/models/finetuned \
  --language en \
  --task transcribe \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --learning-rate 1e-4 \
  --warmup-steps 100 \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --val-split 0.1 \
  --num-beams 5 \
  --seed 42
```

### 5.2 — Every flag, explained

| Flag | Default | What it does | When to touch it |
|---|---|---|---|
| `--data` | *(required)* | Manifest path: JSONL, CSV, or a directory of wav+txt pairs | — |
| `--base-model` | `openai/whisper-base` | HF id or local dir of the model to adapt. `tiny` = fast smoke, `small`/`medium` = more accuracy | Use a local path for air-gapped boxes |
| `--output-dir` | `data/models/finetuned` | Where `adapter/`, `report_*.json` and `ct2/` are written | — |
| `--language` | auto-detect | Pin `en` to skip per-utterance detection and speed things up | Always pin when you know the language |
| `--task` | `transcribe` | `transcribe` \| `translate` | Leave default (bank use is transcription) |
| `--epochs` | `3` | Passes over the training set | Small corpora overfit fast — watch val WER (Step 6) |
| `--batch-size` | `4` | Examples per optimizer step | Reduce to 1 on low RAM; raise on GPU |
| `--grad-accum-steps` | `1` | Effective batch = `batch-size × grad-accum-steps` | Raise to 4 if you must lower batch-size |
| `--learning-rate` | `1e-4` | AdamW LR | LoRA is stable around 1e-4–2e-4; raise cautiously to 2e-4 |
| `--warmup-steps` | `100` | Linear-warmup steps before full LR | Fine as-is for most runs |
| `--max-steps` | `-1` | Hard cap on optimizer steps (`-1` = all epochs) | `--max-steps 100` for quick experiments |
| `--lora-r` | `8` | LoRA rank — capacity of the adapters | 4 = lighter, 16 = more capacity |
| `--lora-alpha` | `32` | LoRA scaling (`alpha/r` effective scale) | Keep ~4× `r` |
| `--lora-dropout` | `0.05` | Dropout on adapters (regularization) | Raise to 0.1 if val WER > train WER |
| `--val-split` | `0.1` | Fraction held out for baseline/finetuned WER | Keep ≥ 0.05 even on small sets (min 1 utterance) |
| `--baseline-limit` | `20` | Max utterances used for the WER probes | Raise for a more stable estimate on bigger data |
| `--num-beams` | `5` | Beam width for the WER probes | Match your serving `stt.beam_size` (default 5) |
| `--max-audio-seconds` | `30.0` | Utterances longer than this are truncated | Only lower it |
| `--seed` | `42` | RNG seed for the train/val split + training | Keep fixed for reproducibility |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` | Let it detect; force `cuda` on a GPU box |
| `--no-export-ct2` | *(off)* | Skip the CTranslate2 conversion (adapter-only run) | For pure experimentation |

### 5.3 — What happens inside, step by step

Walking through `_run()` in `src/speechai/finetune/train.py`:

1. **Seeding** — `_set_seed(42)`: Python, NumPy, PyTorch, and CUDA RNGs all
   fixed so runs are reproducible.
2. **Device resolution** — `auto` → CUDA if available, else CPU.
3. **Output dir** created (`data/models/finetuned/`).
4. **Data split** — `split_manifest()` loads the manifest and deterministically
   shuffles with `random.Random(seed)`, holding out `val_split` (default 10%,
   minimum 1 utterance) for validation.
5. **Model load** — `WhisperProcessor` + `WhisperForConditionalGeneration` load
   from `--base-model` (first run downloads from the HF Hub).
6. **Dataset build** — `WhisperDataset` converts every utterance to log-mel
   features + tokenized labels (Step 2.4).
7. **Baseline WER probe** — the *stock* model decodes up to
   `--baseline-limit` validation utterances with `num_beams=5`; WER/CER/RTF are
   computed with `jiwer` and written to `report_baseline.json`.
8. **LoRA attach** — `LoraConfig(r=8, alpha=32, dropout=0.05, target_modules=["q_proj","v_proj"], task_type="SEQ_2_SEQ_LM")`.
   Only the attention **query/value projection** matrices get adapters — the
   standard Whisper PEFT recipe. The console prints
   `trainable params: X / Y (Z%)` — expect well under 3%.
9. **Training loop** — AdamW (`lr=1e-4`) with a linear-warmup-then-decay
   schedule; gradient clipping at norm 1.0; optional grad accumulation. Loss is
   cross-entropy over the reference tokens (padding masked with `-100`).
   Progress logs every 10 steps (hard-coded `log_every_steps` in `TrainConfig`).
10. **Fine-tuned WER probe** — same validation set through the *adapted* model;
    written to `report_finetuned.json`.
11. **Persist adapter** — `model.save_pretrained(<out>/adapter)` +
    processor to `<out>/`.
12. **Export CT2** (unless `--no-export-ct2`) — merge adapters into the base
    weights (`merge_and_unload`), then `TransformersConverter.convert()` with
    `quantization="int8"` to `<out>/ct2/`, and copy `tokenizer.json`,
    `preprocessor_config.json`, `generation_config.json` next to `model.bin`
    so faster-whisper can load the directory as a standalone model.

### 5.4 — Expected console output (abridged real run)

```text
=== Bank Speech AI - Whisper LoRA fine-tuning ===
base model : openai/whisper-base
device     : cpu
output dir : data/models/finetuned

manifest    : data/manifest.jsonl (120 utterances)
train/val   : 108 / 12

loading processor + base model (first run downloads weights)...

probing baseline WER on 12 val utterances...
BASELINE WER  WER 0.412  CER 0.210  RTF 0.34  (n=12)
trainable params: 8,912,896 / 122,880,000 (7.254%)

training (3 epochs, batch 4, lr 0.0001, grad_accum 1)...
step 10  loss 0.3012  lr 8.33e-05
step 20  loss 0.2130  lr 1.00e-04
...
epoch 1 complete  mean loss 0.2511
...
training finished in 342.1s (81 optimizer steps)

evaluating fine-tuned model...
FINE-TUNED WER  WER 0.183  CER 0.091  RTF 0.40  (n=12)
improvement: baseline 0.412 -> finetuned 0.183

adapter + processor saved to data/models/finetuned
CTranslate2 model exported to: data/models/finetuned/ct2
swap the platform to the fine-tuned model with:
    SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2

Done.
```

That `0.412 → 0.183` WER improvement on the held-out split is the whole point
of the exercise.

---

## Step 6 — Read and interpret the results

Two reports are written into `--output-dir`:

| File | Contents |
|---|---|
| `report_baseline.json` | Per-utterance WER/CER/RTF of the **stock** model on the validation split |
| `report_finetuned.json` | Same metrics for the **adapted** model |

Each contains `aggregates` (`mean`, `median`, `p90`) and per-utterance
`results` (audio path, reference, hypothesis, wer, cer, rtf).

**How to read them:**

1. **Look at the hypothesis vs reference diffs** — the biggest WER rows show you
   *exactly* what the model still gets wrong. Are they words you never included
   in training? → collect more data for that phrase.
2. **Compare WER means** — your goal: finetuned WER << baseline WER on held-out
   data. If it's flat, see Troubleshooting #T4.
3. **Watch for overfitting** — if training loss keeps dropping but val WER
   stalls or rises, you're memorizing. Fix: fewer epochs, more data, or higher
   `--lora-dropout`.
4. **RTF sanity** — RTF should stay well below 1.0 (real-time). The int8
   CTranslate2 export generally runs *faster* than the PyTorch probe, so don't
   worry if probe RTF looks a bit high.

---

## Step 7 — Swap the fine-tuned model into the platform

The faster-whisper engine (`WhisperSTTEngine._model_ref`) loads
`stt.model_path` **if set**, otherwise `stt.model_size`. So swapping is a
single config change — no code changes, no rebuild.

### Option A — environment variable (session-scoped, great for testing)

```bash
export SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2
speechai models          # should now show the model path
speechai transcribe data/samples/sample_01_account_balance.wav --json
```

### Option B — config file (persistent)

`configs/config.yaml`:

```yaml
stt:
  model_size: base        # fallback when model_path is empty
  model_path: data/models/finetuned/ct2   # takes precedence
```

Then restart the API. In **Docker** the `./data` folder is bind-mounted to
`/data`, so use:

```bash
export SPEECHAI_STT__MODEL_PATH=/data/models/finetuned/ct2   # inside docker-compose environment
# or in docker-compose.yml under api/worker:
#   environment: [..., "SPEECHAI_STT__MODEL_PATH=/data/models/finetuned/ct2"]
```

### Verify the swap end to end

```bash
# CLI:
speechai models
speechai transcribe data/samples/sample_04_fraud_alert.wav --json

# REST (if the API is running):
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@data/samples/sample_04_fraud_alert.wav" \
  -F "vad_filter=false"

# Health/status:
curl http://localhost:8000/v1/models   # stt.config.model_path shows the CT2 dir
```

All the platform behaviors still apply unchanged: sentence-level segment
regrouping, PII redaction, VAD options, streaming — they operate on whatever
model `stt.model_path` points at.

> **Rollback is trivial:** clear `model_path` (or unset the env var) and
> restart — the platform falls back to `model_size` (the stock model).

---

## Step 8 — Evaluate and gate the new model

Run the full evaluation harness over your held-out/eval manifest:

```bash
speechai evaluate data/manifest.jsonl --gate
```

- Prints a WER/CER/RTF table (mean / median / p90 + worst-5-by-WER).
- Exports JSON to `data/eval/<dataset>-<engine>.json`.
- `--gate` **fails the command (non-zero exit)** if mean WER > 0.10 or mean RTF
  > 0.50 (`eval.default_wer_tolerance` / `eval.default_rtf_tolerance` in
  `configs/config.yaml`).

**CI pattern** — add this to your pipeline so future model swaps must hold the
bar:

```bash
# in CI, with SPEECHAI_STT__MODEL_PATH exported:
speechai evaluate data/eval_manifest.jsonl --gate
```

Combine with the training reports for a before/after:

| Metric | Baseline | Fine-tuned |
|---|---|---|
| WER (mean) | 0.412 | 0.183 |
| CER (mean) | 0.210 | 0.091 |
| RTF (mean) | 0.34 | 0.40 |

Every training run also pushes `stt_wer{finetune-baseline}` /
`stt_wer{finetune-finetuned}` (and CER/RTF) to the Prometheus gauges, so the
same metrics appear in your dashboards (`deploy/prometheus/alerts.yml` can be
extended to alert on regressions).

---

## Step 9 — Production considerations

**9.1 — Training on GPU**

```bash
speechai-finetune --device cuda --batch-size 8 --grad-accum-steps 2 ...
```

For multi-GPU/mixed precision, wrap the script with `accelerate`
(installed with the `finetune` extra):

```bash
accelerate launch -m speechai.finetune.train --data ... --device cuda ...
```

The loop is intentionally a plain PyTorch loop for transparency; accelerate
wraps it without code changes.

**9.2 — Larger corpora (tens of hours of audio)**

`WhisperDataset` builds log-mel features **in memory**. For very large sets,
precompute `input_features` to disk (`*.npy`) once and memory-map them in a
custom dataset — the dataset API (`__getitem__` returning
`{"input_features", "labels", ...}`) makes this a drop-in swap.

**9.3 — Where the export happens**

CTranslate2 conversion (`TransformersConverter` → `model.bin`, int8) is
**CPU-only and takes minutes for `base`**. Run it in a build job, not on the
inference host, and ship the `ct2/` directory as an artifact.

**9.4 — Docker / deployment**

- Keep the model inside `./data` (bind-mounted to `/data` in the containers)
  so it survives rebuilds and is shared by `api` + `worker`.
- `HF_HOME=/data/hf-cache` already persists HuggingFace downloads in Docker —
  this is what `download_models.py` and first-time model loads use.
- The Docker image ships `faster-whisper` + `piper` (the `engines` extra), but
  **not** torch/transformers — training stays a host-side (or CI) job to keep
  the image lean.

**9.5 — Guardrails for banking**

- **Anonymize** the training corpus (remove PII) — see the platform's own
  redaction (`speechai.redaction`) for what the bank considers sensitive.
- **Audit trail:** keep `report_baseline.json` / `report_finetuned.json` and
  the manifest alongside each exported `ct2/` dir (e.g. commit them or store in
  artifact storage keyed by model version).
- **Rollback plan:** never overwrite a working `model_path`; deploy new models
  to a *new* directory (`data/models/finetuned-v2/ct2`) and switch the config.

---

## 14. Troubleshooting

| # | Symptom | Cause / fix |
|---|---|---|
| T1 | `Missing dependency: No module named 'torch'` | Install the extra: `pip install -e ".[finetune]"` (+ CPU torch on Windows) |
| T2 | `transformers` import errors or forward signature errors | You're on transformers ≥ 5 — the extra pins `<5`; reinstall with `pip install "transformers>=4.44,<5"` |
| T3 | Hang / download at "loading processor + base model" | First run downloads the base model from the HF Hub; needs network. Pre-download and pass `--base-model <local dir>` |
| T4 | WER barely improves after training | Data problem, not code: references don't match audio, vocabulary under-represented, wrong `--language`, or too little data. Re-examine the worst-WER rows in `report_finetuned.json` |
| T5 | Val WER rises while train loss falls | Overfitting — fewer epochs, more data, or `--lora-dropout 0.1` |
| T6 | Out of memory during training | Lower `--batch-size 1` and raise `--grad-accum-steps 4`; close other apps; 8 GB RAM minimum |
| T7 | `No examples found in manifest` | Manifest path wrong or empty; check paths relative to your CWD (Step 3 gotcha) |
| T8 | Export step errors / warnings about missing `tokenizer.json` | The converter copies `tokenizer.json`, `preprocessor_config.json`, `generation_config.json` from the base model; if your base model repo lacks one, download them manually into `ct2/` |
| T9 | faster-whisper fails to load the exported dir | Ensure `stt.model_path` points at the directory **containing** `model.bin` (i.e. `.../ct2`), not the parent; check permissions on `./data` mounts |
| T10 | Segments come back as one big block after swapping | Unrelated to fine-tuning — see the sentence-level segmentation feature (inter-word-gap regrouping). Set `vad_filter=false` for non-clean audio |
| T11 | Training extremely slow on CPU | Expected for `base`+; use `--base-model openai/whisper-tiny` for iteration, or a GPU for the real run |
| T12 | `.venv/Scripts/speechai-finetune.exe` not found | Re-run `pip install -e ".[finetune]"`; verify with `python -m speechai.finetune.train --help` |

---

## 15. Reference — every flag, output artifact, and file

### CLI surface

```text
speechai-finetune --data DATA [options]
  --base-model BASE_MODEL        openai/whisper-base (or local dir)
  --output-dir OUTPUT_DIR        data/models/finetuned
  --language LANGUAGE            e.g. en
  --task {transcribe,translate}
  --epochs EPOCHS                3
  --batch-size BATCH_SIZE        4
  --grad-accum-steps N           1
  --learning-rate LR             1e-4
  --warmup-steps N               100
  --max-steps N                  -1 (all epochs)
  --lora-r N                     8
  --lora-alpha N                 32
  --lora-dropout F               0.05
  --val-split F                  0.1
  --baseline-limit N             20
  --num-beams N                  5
  --max-audio-seconds F          30.0
  --seed N                       42
  --device {auto,cpu,cuda}
  --no-export-ct2                skip conversion
```

### Artifacts produced under `--output-dir`

```
<output-dir>/
  adapter/                    # PEFT LoRA adapter weights (safetensors) + config
  preprocessor_config.json    # HF processor saved to the output dir root
  tokenizer.json              # ... together with the other processor files
  report_baseline.json        # stock-model WER/CER/RTF on the val split
  report_finetuned.json       # adapted-model WER/CER/RTF on the val split
  ct2/                        # ← point stt.model_path HERE
    model.bin                 # merged + int8-quantized CTranslate2 model
    model.bin.json            # CT2 config
    tokenizer.json            # faster-whisper requirement
    preprocessor_config.json  # faster-whisper requirement
    generation_config.json    # faster-whisper requirement
```

### Code map

| Concern | File |
|---|---|
| Entry point / training loop / export | `src/speechai/finetune/train.py` |
| Dataset, collate, split | `src/speechai/finetune/dataset.py` |
| Manifest formats | `src/speechai/eval/loader.py` |
| WER/CER/RTF computation | `src/speechai/eval/metrics.py` |
| Model swap (`model_path or model_size`) | `src/speechai/stt/whisper_engine.py` (`_model_ref`) |
| Config key | `configs/config.yaml` → `stt.model_path` |

### Related documents

- [`docs/finetuning.md`](docs/finetuning.md) — concise version of this guide
- [`END-TO-END-FLOW.md`](END-TO-END-FLOW.md) — how the STT engine consumes the model at runtime
- [`QUICKSTART-CHEATSHEET.md`](QUICKSTART-CHEATSHEET.md) — fast start/stop + curl reference
- [`docs/production-checklist.md`](docs/production-checklist.md) — go-live checks

---

*Happy fine-tuning. The one-sentence recipe: collect representative bank audio,
write a manifest, run `speechai-finetune`, point `stt.model_path` at the
exported `ct2/`, and gate the swap with `speechai evaluate --gate`.*
