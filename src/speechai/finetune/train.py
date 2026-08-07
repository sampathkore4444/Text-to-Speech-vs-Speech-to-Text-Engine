"""LoRA fine-tuning of Whisper on bank-domain audio, with CTranslate2 export.

End-to-end recipe (see ``docs/finetuning.md`` for details):

1. Prepare a manifest (the same format the evaluation harness uses):
       {"audio": "data/samples/sample_01.wav", "reference": "Your account balance ..."}

2. Train:
       speechai-finetune --data data/manifest.jsonl --base-model openai/whisper-base \
           --output-dir data/models/finetuned --epochs 3 --language en

   This reports WER *before* training (baseline), trains LoRA adapters on the
   encoder+decoder attention (q_proj, v_proj), reports WER *after*, and exports
   a merged CTranslate2 model into ``<output-dir>/ct2``.

   Runs are crash-safe: a checkpoint (model + optimizer + scheduler + RNG +
   config) is saved at every epoch boundary, and an interrupted run resumes with
   the same flags plus ``--resume-from <output-dir>/checkpoints/latest.pt``.
   ``--patience`` adds early stopping on held-out WER (restores the best model).

3. Swap the platform to the fine-tuned model (no code changes):
       SPEECHAI_STT__MODEL_PATH=data/models/finetuned/ct2
   or set ``stt.model_path`` in ``configs/config.yaml``. The faster-whisper
   engine picks it up via ``stt.model_path`` / ``stt.model_size``.

Requires the ``finetune`` extra: ``pip install -e '.[finetune]'``
(CPU torch on Windows: ``pip install torch --index-url https://download.pytorch.org/whl/cpu``).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from speechai.core import metrics
from speechai.core.timing import compute_rtf
from speechai.core.tracking import ExperimentTracker
from speechai.eval.metrics import EvaluationReport, UtteranceResult, char_error_rate, word_error_rate
from speechai.finetune.dataset import WhisperDataset, collate_whisper_batch, split_manifest

logger = logging.getLogger("speechai.finetune")


@dataclass
class TrainConfig:
    data: str
    base_model: str = "openai/whisper-base"
    output_dir: str = "data/models/finetuned"
    language: str | None = None
    task: str = "transcribe"
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 = train all epochs
    log_every_steps: int = 10
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    val_split: float = 0.1
    baseline_limit: int = 20  # utterances for the baseline WER probe
    num_beams: int = 5
    max_audio_seconds: float = 30.0
    seed: int = 42
    device: str = "auto"
    export_ct2: bool = True
    # Checkpointing & early stopping
    resume_from: str | None = None  # path to a checkpoint to continue from
    save_every_steps: int = 0  # 0 = save at the end of every epoch
    patience: int = 0  # early stop after N val-WER evals without improvement (0 = off)
    min_delta: float = 0.0  # minimum absolute WER improvement to count as progress
    eval_every_epochs: int = 1  # held-out WER probe cadence when --patience is set
    # Optional MLflow experiment tracking (no-op unless a URI is configured)
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "speechai-finetune"
    mlflow_run_name: str | None = None
    mlflow_log_model: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speechai-finetune",
        description="LoRA fine-tune Whisper on bank-domain audio and export a CTranslate2 model.",
    )
    parser.add_argument("--data", required=True, help="Manifest (JSONL/CSV/dir of wav+txt)")
    parser.add_argument("--base-model", default="openai/whisper-base")
    parser.add_argument("--output-dir", default="data/models/finetuned")
    parser.add_argument("--language", default=None, help="e.g. en (auto-detect if unset)")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--baseline-limit", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-export-ct2", action="store_true", help="skip CTranslate2 export")
    parser.add_argument(
        "--resume-from", default=None,
        help="resume training from a checkpoint file "
             "(e.g. <output-dir>/checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--save-every-steps", type=int, default=0,
        help="save a checkpoint every N optimizer steps (0 = end of every epoch)",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="early stop after this many val-WER evaluations without improvement "
             "(0 = disabled)",
    )
    parser.add_argument(
        "--min-delta", type=float, default=0.0,
        help="minimum absolute WER improvement to count as progress (with --patience)",
    )
    parser.add_argument(
        "--eval-every-epochs", type=int, default=1,
        help="evaluate held-out WER every N epochs when --patience is set",
    )
    parser.add_argument(
        "--mlflow-tracking-uri", default=None,
        help="MLflow tracking server URI (env MLFLOW_TRACKING_URI is also honored)",
    )
    parser.add_argument(
        "--mlflow-experiment", default="speechai-finetune",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-run-name", default=None,
        help="MLflow run name (default: auto)",
    )
    parser.add_argument(
        "--mlflow-log-model", action="store_true",
        help="upload the exported ct2 model directory as an MLflow artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    config = TrainConfig(
        data=args.data,
        base_model=args.base_model,
        output_dir=args.output_dir,
        language=args.language,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        val_split=args.val_split,
        baseline_limit=args.baseline_limit,
        num_beams=args.num_beams,
        max_audio_seconds=args.max_audio_seconds,
        seed=args.seed,
        device=args.device,
        export_ct2=not args.no_export_ct2,
        resume_from=args.resume_from,
        save_every_steps=args.save_every_steps,
        patience=args.patience,
        min_delta=args.min_delta,
        eval_every_epochs=args.eval_every_epochs,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_log_model=args.mlflow_log_model,
    )
    tracker = ExperimentTracker(
        enabled=True,
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment,
        run_name=config.mlflow_run_name,
    )
    try:
        _run(config, tracker)
    except ImportError as exc:
        print(f"\nMissing dependency: {exc}")
        print("Install the finetune extra:  pip install -e '.[finetune]'")
        print("(CPU torch on Windows:  pip install torch --index-url https://download.pytorch.org/whl/cpu)")
        tracker.end(status="FAILED")
        return 2
    except Exception as exc:  # surface clean failures
        logger.exception("fine-tuning failed")
        print(f"\nERROR: {exc}")
        tracker.end(status="FAILED")
        return 1
    tracker.end()
    return 0


# ---------------------------------------------------------------------------
def _run(config: TrainConfig, tracker: ExperimentTracker) -> None:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Bank Speech AI - Whisper LoRA fine-tuning ===")
    print(f"base model : {config.base_model}")
    print(f"device     : {device}")
    print(f"output dir : {output_dir}")

    tracker.start(tags={"task": "finetune", "base_model": config.base_model})
    # Log the full run config, but never the mlflow_* keys themselves (they
    # could contain a credentials-bearing tracking URI).
    tracker.log_params(
        {
            key: value
            for key, value in dataclasses.asdict(config).items()
            if not key.startswith("mlflow_")
        }
    )
    if tracker.enabled:
        uri = config.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
        print(f"mlflow     : tracking to {uri} (experiment {config.mlflow_experiment})")

    # 1. Data -----------------------------------------------------------------
    train_examples, val_examples = split_manifest(config.data, val_split=config.val_split, seed=config.seed)
    print(f"\nmanifest    : {config.data} ({len(train_examples) + len(val_examples)} utterances)")
    print(f"train/val   : {len(train_examples)} / {len(val_examples)}")

    print("\nloading processor + base model (first run downloads weights)...")
    processor = WhisperProcessor.from_pretrained(config.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    model.to(device)

    train_dataset = WhisperDataset(train_examples, processor, max_audio_seconds=config.max_audio_seconds)
    val_dataset = WhisperDataset(val_examples, processor, max_audio_seconds=config.max_audio_seconds)

    # 2. Baseline WER (base model, before any adaptation) ----------------------
    baseline = None
    if config.resume_from is not None:
        print("resuming run - skipping the baseline probe (already recorded on first start)")
    elif val_dataset.items:
        print(f"\nprobing baseline WER on {min(len(val_dataset), config.baseline_limit)} val utterances...")
        baseline = _evaluate_wer(
            model, processor, val_dataset,
            language=config.language, task=config.task,
            num_beams=config.num_beams, device=device, limit=config.baseline_limit,
            engine=f"{config.base_model} (baseline)",
        )
        _save_report(baseline, output_dir / "report_baseline.json", tag="baseline")
        print(_report_line(baseline, "BASELINE WER"))
        tracker.log_metrics(
            {
                "baseline_wer": baseline.aggregates["wer"]["mean"],
                "baseline_cer": baseline.aggregates["cer"]["mean"],
                "baseline_rtf": baseline.aggregates["rtf"]["mean"],
            }
        )

    # 3. Attach LoRA -----------------------------------------------------------
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.3f}%)")
    model.to(device)

    # 4. Train ------------------------------------------------------------------
    dataloader = _make_dataloader(train_dataset, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    steps_per_epoch = max(len(dataloader) // config.grad_accum_steps, 1)
    total_steps = (
        config.max_steps if config.max_steps > 0 else steps_per_epoch * config.epochs
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Resume support: a checkpoint restores model/optimizer/scheduler state and
    # the RNG, so an interrupted run continues instead of restarting.
    start_epoch = 1
    global_step = 0
    best_val_wer: float | None = None
    best_state_path: Path | None = None
    checkpoints_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoints_dir / "latest.pt"
    if config.resume_from:
        start_epoch, global_step, best_val_wer = _load_checkpoint(
            config.resume_from,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        # Carry the pre-interruption early-stopping state forward.
        if config.patience > 0 and best_val_wer is not None:
            best_state_path = checkpoints_dir / "best.pt"
        detail = f", best val WER {best_val_wer:.4f}" if best_val_wer is not None else ""
        print(
            f"resumed from {config.resume_from} "
            f"(continuing at epoch {start_epoch}, step {global_step}{detail})"
        )

    print(f"\ntraining ({config.epochs} epochs, batch {config.batch_size}, "
          f"lr {config.learning_rate}, grad_accum {config.grad_accum_steps})...")
    model.train()
    optimizer.zero_grad()
    if not config.resume_from:
        # Baseline checkpoint so a crash before the first epoch save is resumable.
        _save_checkpoint(
            checkpoint_path, config=config, model=model, optimizer=optimizer,
            scheduler=scheduler, next_epoch=start_epoch, global_step=global_step,
            best_val_wer=best_val_wer,
        )

    use_early_stop = config.patience > 0 and bool(val_dataset.items)
    if config.patience > 0 and not val_dataset.items:
        print("warning: no validation utterances - early stopping disabled")

    stall_count = 0
    start_time = time.perf_counter()
    trained = False
    for epoch in range(start_epoch, config.epochs + 1):
        trained = True
        epoch_loss = 0.0
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = _forward(model, input_features=input_features, labels=labels)
            loss = outputs.loss / config.grad_accum_steps
            loss.backward()
            epoch_loss += float(outputs.loss.item())

            if (global_step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            global_step += 1

            if global_step % config.log_every_steps == 0:
                logger.info(
                    "step %d  loss %.4f  lr %.2e",
                    global_step,
                    outputs.loss.item(),
                    scheduler.get_last_lr()[0],
                )
            if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                _save_checkpoint(
                    checkpoint_path, config=config, model=model, optimizer=optimizer,
                    scheduler=scheduler, next_epoch=epoch, global_step=global_step,
                    best_val_wer=best_val_wer,
                )
            if config.max_steps > 0 and global_step >= config.max_steps:
                break
        logger.info("epoch %d complete  mean loss %.4f", epoch, epoch_loss / max(len(dataloader), 1))
        tracker.log_metrics(
            {
                "train_loss": epoch_loss / max(len(dataloader), 1),
                "lr": scheduler.get_last_lr()[0],
            },
            step=global_step,
        )
        if config.save_every_steps == 0:
            _save_checkpoint(
                checkpoint_path, config=config, model=model, optimizer=optimizer,
                scheduler=scheduler, next_epoch=epoch + 1, global_step=global_step,
                best_val_wer=best_val_wer,
            )
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

        # Early stopping: probe held-out WER and stop when it plateaus, restoring
        # the best model seen so far.
        if use_early_stop and epoch % config.eval_every_epochs == 0:
            report = _evaluate_wer(
                model, processor, val_dataset,
                language=config.language, task=config.task,
                num_beams=config.num_beams, device=device, limit=config.baseline_limit,
                engine=f"early-stop (epoch {epoch})",
            )
            if not report.results:
                print(f"epoch {epoch}  val WER probe returned no results - skipping")
                continue
            wer = float(report.aggregates["wer"]["mean"])
            tracker.log_metrics({"val_wer": wer}, step=global_step)
            best_val_wer, stall_count, should_stop, improved = _early_stop_update(
                best_val_wer, wer, min_delta=config.min_delta,
                patience=config.patience, stall_count=stall_count,
            )
            print(
                f"epoch {epoch}  val WER {wer:.4f}  best {best_val_wer:.4f}  "
                f"(stall {stall_count}/{config.patience})"
            )
            if improved:
                best_state_path = checkpoints_dir / "best.pt"
                torch.save(model.state_dict(), best_state_path)
            if should_stop:
                logger.info(
                    "early stopping: val WER not improved for %d evaluation(s)",
                    config.patience,
                )
                if best_state_path is not None:
                    best_state = torch.load(
                        best_state_path,
                        map_location=torch.device(device),
                        weights_only=False,
                    )
                    model.load_state_dict(best_state)
                    print(f"restored best model (val WER {best_val_wer:.4f})")
                break
    if not trained:
        print(f"checkpoint already trained through epoch {config.epochs} - skipping training")
    elapsed = time.perf_counter() - start_time
    print(f"\ntraining finished in {elapsed:.1f}s ({global_step} optimizer steps)")

    # 5. Fine-tuned WER ----------------------------------------------------------
    if val_dataset.items:
        print("evaluating fine-tuned model...")
        finetuned = _evaluate_wer(
            model, processor, val_dataset,
            language=config.language, task=config.task,
            num_beams=config.num_beams, device=device, limit=config.baseline_limit,
            engine="whisper + LoRA",
        )
        _save_report(finetuned, output_dir / "report_finetuned.json", tag="finetuned")
        print(_report_line(finetuned, "FINE-TUNED WER"))
        improvement = (
            baseline.aggregates["wer"]["mean"] - finetuned.aggregates["wer"]["mean"]
            if baseline is not None
            else None
        )
        tracker.log_metrics(
            {
                "wer": finetuned.aggregates["wer"]["mean"],
                "cer": finetuned.aggregates["cer"]["mean"],
                "rtf": finetuned.aggregates["rtf"]["mean"],
                "improvement": improvement,
            }
        )
        if baseline is not None:
            print(f"improvement: baseline {baseline.aggregates['wer']['mean']:.3f} -> "
                  f"finetuned {finetuned.aggregates['wer']['mean']:.3f}")

    # 6. Persist adapter + export CTranslate2 -------------------------------------
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(output_dir))
    print(f"adapter + processor saved to {output_dir}")

    if config.export_ct2:
        ct2_dir = _export_ct2(model, processor, config.base_model, output_dir)
        print(f"\nCTranslate2 model exported to: {ct2_dir}")
        print("swap the platform to the fine-tuned model with:")
        print(f"    SPEECHAI_STT__MODEL_PATH={ct2_dir}")
        print("(or set stt.model_path in configs/config.yaml)")

    if tracker.enabled:
        for name in ("report_baseline.json", "report_finetuned.json"):
            report_file = output_dir / name
            if report_file.is_file():
                tracker.log_artifact(report_file)
        ct2_dir = output_dir / "ct2"
        if config.mlflow_log_model and ct2_dir.is_dir():
            tracker.log_artifact(ct2_dir)

    print("\nDone.")



# ---------------------------------------------------------------------------
def _evaluate_wer(model, processor, dataset, *, language, task, num_beams, device, limit, engine) -> EvaluationReport:
    import torch

    model.eval()
    items = dataset.items[:limit]
    results: list[UtteranceResult] = []
    with torch.no_grad():
        for item in items:
            features = item["input_features"].unsqueeze(0).to(device)
            start = time.perf_counter()
            # Keyword form is required for PEFT-wrapped models (positional args
            # are rejected) and works for the plain base model too.
            generated = model.generate(input_features=features, num_beams=num_beams)
            hypothesis = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            latency = time.perf_counter() - start
            duration = item["duration_seconds"]
            reference = item["reference"]
            results.append(
                UtteranceResult(
                    audio=item["audio"],
                    reference=reference,
                    hypothesis=hypothesis,
                    wer=word_error_rate(reference, hypothesis),
                    cer=char_error_rate(reference, hypothesis),
                    audio_duration=duration,
                    latency_seconds=latency,
                    rtf=compute_rtf(latency, duration),
                )
            )
    model.train()  # back to training mode for the caller
    return EvaluationReport(engine=engine, dataset="finetune-val", results=results)


def _forward(model, **kwargs):
    """Forward through the (possibly PEFT-wrapped) model.

    PEFT's Seq2Seq wrapper injects ``input_ids``/``inputs_embeds`` kwargs that
    Whisper's forward does not accept; the LoRA tuners live in ``base_model``,
    so we fall back to it when the wrapper rejects the call.
    """
    try:
        return model(**kwargs)
    except TypeError:
        base = getattr(model, "base_model", None)
        if base is None:
            raise
        return base(**kwargs)


def _make_dataloader(dataset, config: TrainConfig, device: str):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_whisper_batch,
        num_workers=0,
        pin_memory=device == "cuda",
    )


# ---------------------------------------------------------------------------
def _early_stop_update(best_wer, current_wer, *, min_delta, patience, stall_count):
    """Early-stopping decision helper (pure, unit-testable).

    Returns ``(new_best_wer, new_stall_count, should_stop, improved)``:
    - ``improved``: ``current_wer`` beats the best by at least ``min_delta``.
    - ``should_stop``: WER has not improved for ``patience`` evaluations.
    """
    improved = best_wer is None or current_wer <= best_wer - min_delta
    if improved:
        return current_wer, 0, False, True
    stall = stall_count + 1
    return best_wer, stall, stall >= patience, False


def _save_checkpoint(
    path: Path,
    *,
    config: TrainConfig,
    model,
    optimizer,
    scheduler,
    next_epoch: int,
    global_step: int,
    best_val_wer: float | None = None,
) -> None:
    """Persist a resumable training checkpoint (atomic write).

    Stores the model/optimizer/scheduler state, the RNG state, the current best
    val WER (for early stopping) and the config so a run interrupted at any
    epoch boundary can continue with ``--resume-from``. The payload is a plain
    dict - ``torch.load`` must use ``weights_only=False`` because it contains
    non-tensor RNG state.
    """
    import random

    import numpy as np
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = {
        "config": dataclasses.asdict(config),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "best_val_wer": best_val_wer,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _load_checkpoint(
    path: str | Path,
    *,
    config: TrainConfig,
    model,
    optimizer,
    scheduler,
    device: str,
) -> tuple[int, int]:
    """Load a checkpoint produced by :func:`_save_checkpoint`.

    Validates that the checkpoint was trained with the same model-defining
    configuration, then restores model/optimizer/scheduler and RNG state.
    Returns ``(next_epoch, global_step, best_val_wer)``.
    """
    import random

    import numpy as np
    import torch

    ckpt = torch.load(path, map_location=torch.device(device), weights_only=False)
    saved_config = ckpt.get("config") or {}
    for key in ("base_model", "language", "task", "seed",
                "lora_r", "lora_alpha", "lora_dropout"):
        saved = saved_config.get(key)
        current = getattr(config, key)
        if str(saved) != str(current):
            raise ValueError(
                f"Checkpoint {path} was trained with {key}={saved!r} but this run has "
                f"{current!r}. Resume with the same flags (or use a different "
                f"--output-dir)."
            )
    if not _same_manifest(saved_config.get("data"), config.data):
        raise ValueError(
            f"Checkpoint {path} was trained on {saved_config.get('data')!r} but this run "
            f"uses {config.data!r}. Resume with the same --data."
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    # Restore RNG state so the dataloader shuffle continues where it left off.
    torch.set_rng_state(ckpt["torch_rng"])
    cuda_rng = ckpt.get("cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
    if "numpy_rng" in ckpt:
        np.random.set_state(ckpt["numpy_rng"])
    if "python_rng" in ckpt:
        random.setstate(ckpt["python_rng"])
    return int(ckpt["next_epoch"]), int(ckpt["global_step"]), ckpt.get("best_val_wer")


def _same_manifest(saved, current) -> bool:
    """Compare manifest paths tolerating relative vs. absolute spelling."""
    saved, current = str(saved), str(current)
    try:
        return Path(saved).resolve() == Path(current).resolve()
    except OSError:
        return saved == current


def _export_ct2(peft_model, processor, base_model_name: str, output_dir: Path) -> Path:
    """Merge LoRA into the base weights and convert to a CTranslate2 model
    that faster-whisper can load directly."""
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import hf_hub_download

    logger.info("merging LoRA adapters into base weights...")
    merged = peft_model.merge_and_unload()
    merged.eval()

    # ctranslate2's TransformersConverter takes a *path* to saved transformers
    # weights (it loads them itself), not a model object - persist the merged
    # model first, then convert from disk. The converter also builds the CT2
    # vocabularies from the tokenizer, so the source dir needs the processor
    # files (vocab.json / merges.txt / tokenizer_config.json) next to the
    # weights.
    source_dir = output_dir / "ct2_source"
    merged.save_pretrained(str(source_dir))
    processor.save_pretrained(str(source_dir))

    ct2_dir = output_dir / "ct2"
    # Start clean: the converter refuses to overwrite an existing directory
    # (on some versions --force only exists on the CLI wrapper, and a stale
    # dir from a failed run would otherwise abort the export).
    shutil.rmtree(ct2_dir, ignore_errors=True)
    ct2_dir.mkdir(parents=True, exist_ok=True)
    logger.info("converting to CTranslate2 (a few minutes on CPU)...")
    converter = TransformersConverter(str(source_dir), copy_files=[])
    # ctranslate2 3.x used convert_to_file(); 4.x renamed it to convert().
    convert = getattr(converter, "convert", None) or converter.convert_to_file
    try:
        convert(str(ct2_dir), quantization="int8", force=True)
    except TypeError:  # older versions lack these kwargs
        try:
            convert(str(ct2_dir), quantization="int8")
        except TypeError:
            convert(str(ct2_dir))
    # The fp32 source weights are only needed for the conversion - drop them
    # so the output dir carries just the artifacts the platform serves.
    shutil.rmtree(source_dir, ignore_errors=True)

    # faster-whisper needs the tokenizer + preprocessor next to model.bin.
    # Prefer the processor files saved into the source dir - that works even
    # for local/air-gapped base models - and only fall back to the HF hub.
    for name in ("tokenizer.json", "preprocessor_config.json", "generation_config.json"):
        local_file = source_dir / name
        if local_file.is_file():
            shutil.copy(local_file, ct2_dir / name)
            logger.info("copied %s from merged source", name)
            continue
        try:
            hub_path = hf_hub_download(base_model_name, name)
            shutil.copy(hub_path, ct2_dir / name)
            logger.info("copied %s from base model", name)
        except Exception:
            logger.warning("could not copy %s (missing in source dir and base model)", name)
    return ct2_dir


def _save_report(report: EvaluationReport, path: Path, *, tag: str) -> None:
    report.export_json(path)
    metrics.stt_wer.labels(f"finetune-{tag}").set(report.aggregates["wer"]["mean"])
    metrics.stt_cer.labels(f"finetune-{tag}").set(report.aggregates["cer"]["mean"])
    metrics.stt_rtf_mean.labels(f"finetune-{tag}").set(report.aggregates["rtf"]["mean"])


def _report_line(report: EvaluationReport, label: str) -> str:
    agg = report.aggregates
    return (
        f"{label:>22}  WER {agg['wer']['mean']:.3f}  CER {agg['cer']['mean']:.3f}  "
        f"RTF {agg['rtf']['mean']:.2f}  (n={len(report.results)})"
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return requested


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    sys.exit(main())
