# Fine-tuning Whisper with LoRA for bank-domain speech

The platform's STT engine runs Whisper via **faster-whisper** (CTranslate2) for
fast CPU inference. LoRA fine-tuning adapts Whisper to a bank's vocabulary and
acoustic conditions (product names, account jargon, accents, telephony audio)
without touching the frozen base weights.

The workflow is:

```
manifest (audio + reference)  →  PyTorch LoRA training  →  merge adapters
    →  convert to CTranslate2  →  faster-whisper engine (stt.model_path swap)
```

## 1. Install

```bash
pip install -e ".[dev,engines,finetune]"
# CPU-only torch on Windows (avoid the multi-GB CUDA build):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Prepare data

Use the same manifest format as the evaluation harness — JSONL, CSV, or a
directory of `wav + txt` pairs:

```jsonl
{"audio": "data/samples/sample_01_account_balance.wav", "reference": "Your account balance is one thousand two hundred and fifty dollars and fifty cents as of today."}
{"audio": "data/calls/call_0421.wav", "reference": "We detected unusual activity on your account."}
```

> **Volume guidance:** LoRA adapts with as little as 1–2 hours of transcribed
> audio, but the more representative the data (real call audio, same mics/IVR
> codec, full vocabulary coverage) the better the gain. Start with
> `scripts/make_sample_audio.py` + your own recordings.

## 3. Train

```bash
speechai-finetune \
    --data data/manifest.jsonl \
    --base-model openai/whisper-base \
    --output-dir data/models/finetuned \
    --language en \
    --epochs 3 --batch-size 4 --learning-rate 1e-4 \
    --lora-r 8 --lora-alpha 32
```

The script:

1. probes **baseline WER** on the validation split with the un-adapted model,
2. attaches **LoRA** (`q_proj`, `v_proj` — the standard Whisper PEFT recipe),
3. trains (AdamW + linear-warmup schedule, grad clipping, grad accumulation),
4. re-evaluates **WER / CER / RTF** after adaptation,
5. saves the adapter + processor, then **merges and converts** to a CTranslate2
   model (`model.bin` + `tokenizer.json` + `preprocessor_config.json`), and
6. exports both reports (`report_baseline.json`, `report_finetuned.json`) and
   pushes the aggregates to Prometheus gauges (`stt_wer{eval_set="finetune-*"}`).

### Key options

| Flag | Default | Notes |
|---|---|---|
| `--base-model` | `openai/whisper-base` | `tiny` for fast smoke tests, `small/medium` for accuracy |
| `--language` | auto | pin `en` to skip detection |
| `--epochs` | 3 | small corpora overfit fast; watch the val WER |
| `--batch-size` / `--grad-accum-steps` | 4 / 1 | effective batch = product |
| `--learning-rate` | 1e-4 | LoRA is stable around 1e-4–2e-4 |
| `--max-steps` | -1 | `--max-steps 100` for a quick smoke run |
| `--val-split` | 0.1 | held-out WER |
| `--no-export-ct2` | - | skip conversion (adapter-only) |
| `--resume-from` | - | continue an interrupted run from `<output-dir>/checkpoints/latest.pt` |
| `--save-every-steps` | 0 (epoch end) | checkpoint cadence for crash-safe resume |
| `--patience` | 0 (off) | early-stop after N val-WER evals without improvement |
| `--min-delta` | 0.0 | min absolute WER gain to count as progress |
| `--eval-every-epochs` | 1 | held-out WER probe cadence when `--patience` is set |
| `--mlflow-tracking-uri` | - | enable MLflow tracking (env `MLFLOW_TRACKING_URI` honored) |
| `--mlflow-experiment` | `speechai-finetune` | MLflow experiment name |
| `--mlflow-log-model` | off | upload the exported `ct2/` dir as an artifact |

**Checkpointing & early stopping.** A resumable checkpoint (model + optimizer
+ scheduler + RNG + config) is written at the end of every epoch (or every
`--save-every-steps` steps) to `<output-dir>/checkpoints/latest.pt`. Rerun the
same command with `--resume-from <output-dir>/checkpoints/latest.pt` to
continue an interrupted run — the model-defining config is validated and the
baseline probe is skipped. For overfit-prone corpora add
`--patience 3 --min-delta 0.01`: the loop probes held-out WER each epoch and
stops (restoring the best model) once WER stops improving.

**Experiment tracking (MLflow).** Optional and best-effort: pass
`--mlflow-tracking-uri <uri>` (or set `MLFLOW_TRACKING_URI`) to record the run
— every training parameter, `baseline_wer/cer/rtf`, per-epoch
`train_loss`/`lr`/`val_wer`, the final `wer/cer/rtf` + `improvement`, and both
JSON reports (plus the `ct2/` model with `--mlflow-log-model`). Install the
extra with `pip install -e ".[finetune,mlflow]"`. `speechai evaluate` tracks
the same way via the `tracking` section of `configs/config.yaml`.

## 4. Swap the platform to the fine-tuned model

One config change — the faster-whisper engine already supports loading a local
converted directory:

```bash
export SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2
```

or in `configs/config.yaml`:

```yaml
stt:
  model_size: base        # fallback
  model_path: data/models/finetuned/ct2   # takes precedence
```

Verify with:

```bash
speechai models
speechai transcribe data/samples/sample_01_account_balance.wav --json
```

**Verify the export before swapping.** The training probes measure the **fp32
PyTorch model**; what you serve is the **int8 CTranslate2 export** and
quantization can shift accuracy. Check the actual artifact with
`scripts/verify_ct2_model.py` — it evaluates the int8 model through the
platform's own engine + eval harness, compares mean WER against
`report_finetuned.json`, and fails (exit 1) on the quantization gap
(`--max-wer-gap`, default 0.05) or the absolute WER/RTF bars:

```bash
python scripts/verify_ct2_model.py \
  --ct2-dir data/models/finetuned/ct2 \
  --manifest data/eval_manifest.jsonl \
  --fp32-report data/models/finetuned/report_finetuned.json
```

## 5. Evaluate & gate

```bash
speechai evaluate data/manifest.jsonl --gate
```

Compare `report_baseline.json` vs `report_finetuned.json`; the `--gate` flags
WER > 10% / RTF > 0.5 (configurable in `eval.default_*_tolerance`). Add the
evaluation to CI so future model swaps must hold the bar.

## 6. Production notes

- **Conversion** happens on CPU and takes minutes for `base`; do it in a build
  job, not on the inference host.
- **GPU training:** set `--device cuda`. For multi-GPU/mixed precision, wrap the
  training loop with `accelerate` (the extra is installed; the loop is a plain
  torch loop by design for transparency).
- **Larger corpora:** the dataset builds features in memory; for tens of hours
  of audio, precompute `input_features` to disk (`*.npy`) and memory-map them.
- **Serving:** the exported CT2 dir is a normal faster-whisper model — the same
  quantization, beam size and VAD options apply.
- **Overfitting:** small corpora converge in 1–3 epochs; watch train vs val WER.
