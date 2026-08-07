"""Verify an exported int8 CTranslate2 model against the fp32 baseline before swapping.

``speechai-finetune`` evaluates the **fp32 PyTorch model** during training, but
the platform serves the **int8 CTranslate2 export** (``<output-dir>/ct2``) and
quantization can shift accuracy. This script evaluates the exact artifact you
will serve - through the platform's own faster-whisper engine and evaluation
harness - and compares it against the fp32 probe from ``report_finetuned.json``,
failing (non-zero exit) unless the swap gates hold:

- **Quantization gap:**  int8 mean WER - fp32 mean WER <= ``--max-wer-gap``
- **Absolute quality:**  int8 mean WER <= ``--max-wer-abs``
- **Serving speed:**     int8 mean RTF <= ``--max-rtf``

It also **spot-checks the serving paths** - a batch transcription job
(``POST /v1/jobs/transcribe`` -> worker run -> status) and a WebSocket
streaming session (``/v1/ws/transcribe`` with VAD-gated final events) - through
the real FastAPI app in-process, reusing the already-loaded engine so the model
is not loaded twice. A failed spot check blocks the swap (exit 1) just like a
breached gate. Disable either with ``--no-batch-check`` / ``--no-ws-check``.

Usage::

    python scripts/verify_ct2_model.py \\
        --ct2-dir data/models/finetuned/ct2 \\
        --manifest data/eval_manifest.jsonl \\
        --fp32-report data/models/finetuned/report_finetuned.json \\
        --sample-audio data/samples/sample_01_account_balance.wav

Requires only the ``engines`` extra (faster-whisper) - not torch/transformers -
so it runs on the same host that serves the model. Optional MLflow logging via
``--mlflow-tracking-uri`` (or ``--no-mlflow`` to force it off).

Exit codes: 0 = gates pass, 1 = gates breached, 2 = usage/input error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from speechai.core.config import Settings
from speechai.core.tracking import ExperimentTracker
from speechai.eval.metrics import EvaluationReport
from speechai.eval.runner import run_from_manifest
from speechai.stt.base import STTEngine, build_stt_engine

console = Console()


# ---------------------------------------------------------------------------
# Pure logic (unit-testable without an engine or model files)
# ---------------------------------------------------------------------------
def compute_wer_gap(int8_wer_mean: float, fp32_wer_mean: float) -> float:
    """The WER delta introduced by int8 quantization (positive = worse)."""
    return float(int8_wer_mean) - float(fp32_wer_mean)


def gather_problems(
    int8_wer_mean: float,
    int8_rtf_mean: float,
    fp32_wer_mean: float,
    *,
    max_wer_gap: float,
    max_wer_abs: float,
    max_rtf: float,
) -> list[str]:
    """Return every breached swap gate as a human-readable problem list.

    Empty list = the int8 model may be swapped in.
    """
    problems: list[str] = []
    gap = compute_wer_gap(int8_wer_mean, fp32_wer_mean)
    if gap > max_wer_gap:
        problems.append(
            f"quantization gap too large: int8 WER {int8_wer_mean:.4f} - fp32 WER "
            f"{fp32_wer_mean:.4f} = {gap:.4f} > max gap {max_wer_gap}"
        )
    if int8_wer_mean > max_wer_abs:
        problems.append(
            f"int8 WER {int8_wer_mean:.4f} > absolute bar {max_wer_abs}"
        )
    if int8_rtf_mean > max_rtf:
        problems.append(
            f"int8 RTF {int8_rtf_mean:.4f} > serving bar {max_rtf} (model too slow for real time)"
        )
    return problems


def load_fp32_report(path: str | Path) -> dict[str, Any]:
    """Load the training-time fp32 report (``report_finetuned.json``).

    Returns ``{"aggregates": {...}, "utterances": {resolved_audio: result}}``
    where the utterance map key is the resolved audio path.
    """
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"fp32 report not found: {report_path} (run speechai-finetune first, "
            f"then point --fp32-report at <output-dir>/report_finetuned.json)"
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    aggregates = payload.get("aggregates") or {}
    utterances = {
        str(Path(item.get("audio", "")).resolve()): item
        for item in payload.get("utterances") or []
        if item.get("audio")
    }
    return {"aggregates": aggregates, "utterances": utterances}


def top_regressions(
    int8_report: EvaluationReport,
    fp32_utterances: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """The utterances where the int8 model regressed most vs. fp32."""
    rows: list[dict[str, Any]] = []
    for result in int8_report.results:
        fp32 = fp32_utterances.get(str(Path(result.audio).resolve()))
        if fp32 is None:
            continue
        delta = float(result.wer) - float(fp32.get("wer", 0.0))
        if delta > 1e-6:
            rows.append(
                {
                    "audio": result.audio,
                    "reference": result.reference,
                    "fp32_hypothesis": fp32.get("hypothesis", ""),
                    "int8_hypothesis": result.hypothesis,
                    "fp32_wer": round(float(fp32.get("wer", 0.0)), 4),
                    "int8_wer": round(result.wer, 4),
                    "wer_delta": round(delta, 4),
                }
            )
    return sorted(rows, key=lambda r: r["wer_delta"], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Serving-path spot checks (batch job + WebSocket streaming)
# ---------------------------------------------------------------------------
def _resolve_spot_audio(args: argparse.Namespace, report: EvaluationReport) -> tuple[Path, Path | None, bool]:
    """Pick the audio for the serving-path spot checks.

    Preference order: ``--sample-audio``, the first eval manifest utterance,
    else a generated 2 s tone written to a temp dir (returned as the cleanup
    path so the caller can remove it afterwards). The third element flags
    generated audio, which the WS check uses to pick a VAD backend that
    reliably detects synthetic tones.
    """
    if args.sample_audio:
        return Path(args.sample_audio), None, False
    if report.results:
        first = Path(report.results[0].audio)
        if first.is_file():
            return first, None, False
    from speechai.audio.io import generate_sine, write_wav

    temp_dir = Path(tempfile.mkdtemp(prefix="verify_ct2_"))
    write_wav(temp_dir / "tone.wav", generate_sine(2.0, 16000))
    return temp_dir / "tone.wav", temp_dir, True


def _check_batch_path(
    settings: Settings,
    audio_path: str | Path,
    *,
    stt_engine: STTEngine | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Run one transcription job through the real REST batch path.

    Exercises ``POST /v1/jobs/transcribe`` (multipart upload + validation +
    enqueue), the worker step (dequeue -> ``run_job`` -> ``transcribe_sync``)
    and the job status route - with the same already-loaded engine the sync
    eval used (injected so the model is not loaded twice). The app runs on a
    deep copy of the settings with a temp ``data_dir`` so the upload/job
    artifacts the route persists never touch the real data directory.
    """
    from fastapi.testclient import TestClient

    from speechai.api.app import create_app

    try:
        app_settings = settings.model_copy(deep=True)
        app_settings.storage.data_dir = str(tempfile.mkdtemp(prefix="verify_ct2_batch_"))
        cleanup_dir = Path(app_settings.storage.data_dir)
        app = create_app(app_settings, stt_engine=stt_engine)
        try:
            with TestClient(app) as client:
                form = {"language": language} if language else {}
                with open(audio_path, "rb") as fh:
                    response = client.post(
                        "/v1/jobs/transcribe",
                        files={"file": (Path(audio_path).name, fh, "audio/wav")},
                        data=form,
                    )
                if response.status_code != 202:
                    return {
                        "ok": False,
                        "error": f"submit returned HTTP {response.status_code}: {response.text[:200]}",
                    }
                job_id = response.json()["job_id"]

                async def _run_one():
                    job = await client.app.state.pipeline.get_job(job_id)
                    return await client.app.state.pipeline.run_job(job)

                job = asyncio.run(_run_one())
                if job.status.value != "succeeded":
                    return {"ok": False, "error": f"job status {job.status.value}: {job.error}"}
                status_body = client.get(f"/v1/jobs/{job_id}").json()
                result = status_body.get("result") or {}
                return {
                    "ok": True,
                    "job_id": job_id,
                    "status": job.status.value,
                    "text_length": len(result.get("text", "")),
                }
        finally:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _check_ws_path(
    settings: Settings,
    audio_path: str | Path,
    *,
    stt_engine: STTEngine | None = None,
    language: str | None = None,
    energy_vad: bool = False,
) -> dict[str, Any]:
    """Stream audio through the real WebSocket serving path.

    Exercises ``/v1/ws/transcribe``: config message, raw PCM16 chunk streaming,
    VAD utterance segmentation and ``final`` events. Trailing silence is padded
    so the VAD deterministically closes the utterance. ``energy_vad`` is only
    set for generated tones (WebRTC's speech model may reject them) - real
    audio uses the platform's configured backend. ``settings.vad`` is deep-
    copied so the caller's config is untouched.
    """
    from fastapi.testclient import TestClient

    from speechai.api.app import create_app
    from speechai.audio.io import load_audio, pcm16_bytes, to_asr_audio

    try:
        ws_settings = settings.model_copy(deep=True)
        if energy_vad:
            ws_settings.vad.backend = "energy"
        app = create_app(ws_settings, stt_engine=stt_engine)
        with TestClient(app) as client:
            audio = to_asr_audio(load_audio(audio_path))
            lead = np.zeros(int(0.3 * audio.sample_rate), np.float32)
            trail = np.zeros(int(0.8 * audio.sample_rate), np.float32)
            signal = np.concatenate([lead, audio.samples, trail])
            pcm = pcm16_bytes(signal)
            with client.websocket_connect("/v1/ws/transcribe") as ws:
                ws.send_json(
                    {"sample_rate": audio.sample_rate, "language": language or settings.stt.language or "en"}
                )
                for i in range(0, len(pcm), 4800):
                    ws.send_bytes(pcm[i : i + 4800])
                ws.send_json({"action": "stop"})
                events: list[dict[str, Any]] = []
                while True:
                    try:
                        events.append(ws.receive_json())
                    except Exception:
                        break
        errors = [e for e in events if e.get("type") == "error"]
        if errors:
            return {
                "ok": False,
                "error": f"ws error event {errors[0].get('code')}: {errors[0].get('message')}",
            }
        finals = [e for e in events if e.get("type") == "final"]
        if not finals:
            return {"ok": False, "error": "no final event emitted (VAD segmented no speech)"}
        return {
            "ok": True,
            "events": len(events),
            "finals": len(finals),
            "final_text_length": sum(len(e.get("text", "")) for e in finals),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _print_spot_check(name: str, result: dict[str, Any]) -> None:
    if result.get("ok"):
        detail = "  ".join(f"{k}={v}" for k, v in result.items() if k not in ("ok", "error"))
        console.print(f"[green]spot check ({name}): OK[/green]  {detail}")
    else:
        console.print(f"[red]spot check ({name}): FAILED - {result.get('error')}[/red]")


# ---------------------------------------------------------------------------
def build_int8_engine(
    ct2_dir: str | Path,
    settings: Settings,
    *,
    device: str,
    compute_type: str,
) -> STTEngine:
    """Construct the platform's STT engine pointed at the int8 CT2 export.

    Reuses ``Settings`` + ``build_stt_engine`` so the verification exercises
    the exact configuration path the API/CLI serve with. ``model_path`` takes
    precedence over ``model_size`` inside the engine, and we pin the compute
    type to ``int8`` by default so the probe matches the exported quantization.
    """
    settings.stt.model_path = str(ct2_dir)
    settings.stt.device = device
    settings.stt.compute_type = compute_type
    return build_stt_engine(settings)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_ct2_model",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ct2-dir", required=True, help="Exported CTranslate2 dir (contains model.bin)")
    parser.add_argument("--manifest", required=True, help="JSONL/CSV manifest or dir of wav+txt pairs")
    parser.add_argument(
        "--fp32-report", required=True,
        help="Training-time fp32 report: <output-dir>/report_finetuned.json",
    )
    parser.add_argument("--language", default=None, help="e.g. en (auto-detect if unset)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--compute-type", default="int8", choices=["auto", "int8", "int8_float16", "float16", "float32"],
        help="Quantization used to probe the export (default int8 = matches the export)",
    )
    parser.add_argument(
        "--max-wer-gap", type=float, default=0.05,
        help="Max allowed int8 WER - fp32 WER (quantization loss)",
    )
    parser.add_argument(
        "--max-wer-abs", type=float, default=0.10,
        help="Max allowed absolute int8 mean WER",
    )
    parser.add_argument("--max-rtf", type=float, default=0.50, help="Max allowed int8 mean RTF")
    parser.add_argument("--report", default=None, help="Where to write the verification report JSON")
    parser.add_argument(
        "--mlflow-tracking-uri", default=None,
        help="MLflow tracking server URI (env MLFLOW_TRACKING_URI also honored)",
    )
    parser.add_argument("--mlflow-experiment", default="speechai", help="MLflow experiment name")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow experiment tracking")
    parser.add_argument(
        "--sample-audio", default=None,
        help="Audio pushed through the batch/WS spot checks "
             "(default: first manifest utterance, else a generated tone)",
    )
    parser.add_argument(
        "--no-batch-check", action="store_true",
        help="Skip the batch-job serving-path spot check",
    )
    parser.add_argument(
        "--no-ws-check", action="store_true",
        help="Skip the WebSocket streaming serving-path spot check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Default-on serving-path spot checks; --no-batch-check/--no-ws-check opt out.
    args.check_batch = not args.no_batch_check
    args.check_ws = not args.no_ws_check
    settings = Settings.load()

    ct2_dir = Path(args.ct2_dir)
    if not ct2_dir.is_dir() or not (ct2_dir / "model.bin").is_file():
        console.print(
            f"[red]error:[/red] {ct2_dir} is not a converted CTranslate2 model "
            f"(expected {ct2_dir / 'model.bin'}) - point --ct2-dir at <output-dir>/ct2"
        )
        return 2
    if not Path(args.manifest).is_file():
        console.print(f"[red]error:[/red] manifest not found: {args.manifest}")
        return 2

    try:
        fp32 = load_fp32_report(args.fp32_report)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 2

    # Evaluate the int8 export through the platform's own engine + harness.
    engine = build_int8_engine(ct2_dir, settings, device=args.device, compute_type=args.compute_type)
    try:
        report = run_from_manifest(engine, args.manifest, language=args.language)
    except Exception as exc:
        console.print(f"[red]error:[/red] int8 evaluation failed: {exc}")
        engine.close()
        return 1

    int8_agg = report.aggregates
    fp32_agg = fp32["aggregates"]
    int8_wer = float(int8_agg["wer"]["mean"])
    int8_rtf = float(int8_agg["rtf"]["mean"])
    fp32_wer = float((fp32_agg.get("wer") or {}).get("mean", 0.0))
    gap = compute_wer_gap(int8_wer, fp32_wer)
    problems = gather_problems(
        int8_wer, int8_rtf, fp32_wer,
        max_wer_gap=args.max_wer_gap, max_wer_abs=args.max_wer_abs, max_rtf=args.max_rtf,
    )
    regressions = top_regressions(report, fp32.get("utterances", {}))

    # ------------------------------------------------------------------
    # Serving-path spot checks: run the exported model through the batch job
    # and WebSocket streaming routes of the real FastAPI app. The sync-eval
    # engine is injected so the model is loaded exactly once.
    _print_comparison(report, fp32_agg, gap, problems)
    spot_checks: dict[str, Any] = {}
    audio_path, cleanup_dir, is_generated = _resolve_spot_audio(args, report)
    try:
        if args.check_batch:
            spot_checks["batch"] = _check_batch_path(
                settings, audio_path, stt_engine=engine, language=args.language
            )
            _print_spot_check("batch job", spot_checks["batch"])
            if not spot_checks["batch"]["ok"]:
                problems.append(
                    "batch serving path failed: " + str(spot_checks["batch"].get("error"))
                )
        if args.check_ws:
            spot_checks["ws"] = _check_ws_path(
                settings, audio_path, stt_engine=engine, language=args.language,
                energy_vad=is_generated,
            )
            _print_spot_check("WebSocket", spot_checks["ws"])
            if not spot_checks["ws"]["ok"]:
                problems.append(
                    "WebSocket serving path failed: " + str(spot_checks["ws"].get("error"))
                )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    report_path = Path(
        args.report
        or str(settings.eval_report_dir / f"verify_ct2-{report.dataset}-int8.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ct2_dir": str(ct2_dir),
        "manifest": args.manifest,
        "fp32_report": args.fp32_report,
        "engine": report.engine,
        "dataset": report.dataset,
        "created_at": time.time(),
        "n_utterances": len(report.results),
        "compute_type": args.compute_type,
        "int8_aggregates": int8_agg,
        "fp32_aggregates": fp32_agg,
        "wer_gap": round(gap, 4),
        "gates": {
            "max_wer_gap": args.max_wer_gap,
            "max_wer_abs": args.max_wer_abs,
            "max_rtf": args.max_rtf,
        },
        "passed": not problems,
        "problems": problems,
        "top_regressions": regressions,
        "spot_checks": spot_checks,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]verification report written to[/green] {report_path}")

    # Optional MLflow logging (best-effort no-op when disabled). Mirrors
    # ``speechai evaluate``: on via the flag, config ``tracking.enabled``, or
    # the MLFLOW_TRACKING_URI env var; ``--no-mlflow`` forces it off.
    tracking_on = not args.no_mlflow and bool(
        args.mlflow_tracking_uri or settings.tracking.enabled or os.environ.get("MLFLOW_TRACKING_URI")
    )
    tracker = ExperimentTracker(
        enabled=tracking_on,
        tracking_uri=args.mlflow_tracking_uri or settings.tracking.tracking_uri or None,
        experiment_name=args.mlflow_experiment,
        run_name=f"verify-{report.dataset}",
    )
    if tracker.enabled:
        tracker.start(tags={"tool": "verify-ct2", "dataset": report.dataset})
        tracker.log_params(
            {
                "ct2_dir": str(ct2_dir),
                "manifest": args.manifest,
                "fp32_report": args.fp32_report,
                "compute_type": args.compute_type,
                "max_wer_gap": args.max_wer_gap,
                "max_wer_abs": args.max_wer_abs,
                "max_rtf": args.max_rtf,
                "batch_check": args.check_batch,
                "ws_check": args.check_ws,
                "spot_check_batch_ok": spot_checks.get("batch", {}).get("ok"),
                "spot_check_ws_ok": spot_checks.get("ws", {}).get("ok"),
            }
        )
        metrics_: dict[str, float] = {}
        for key in ("wer", "cer", "rtf"):
            for stat in ("mean", "median", "p90"):
                metrics_[f"int8_{key}_{stat}"] = int8_agg[key][stat]
        metrics_["fp32_wer_mean"] = fp32_wer
        metrics_["wer_gap"] = gap
        tracker.log_metrics(metrics_)
        tracker.log_artifact(report_path)
        tracker.end(status="FAILED" if problems else "FINISHED")

    engine.close()
    if problems:
        console.print("[red]verification FAILED - do NOT swap this model in[/red]")
        return 1
    console.print("[green]verification passed - safe to point stt.model_path at this export[/green]")
    return 0


def _print_comparison(
    report: EvaluationReport,
    fp32_agg: dict[str, Any],
    gap: float,
    problems: list[str],
) -> None:
    int8_agg = report.aggregates
    table = Table(title=f"int8 CT2 vs fp32 baseline  ({report.dataset}, n={len(report.results)})")
    table.add_column("Metric", style="bold cyan")
    table.add_column("fp32 (report)", justify="right")
    table.add_column("int8 (served)", justify="right")
    table.add_column("delta", justify="right")
    for key, label in (("wer", "WER"), ("cer", "CER"), ("rtf", "RTF")):
        fp = float((fp32_agg.get(key) or {}).get("mean", 0.0))
        int8 = float(int8_agg[key]["mean"])
        delta = int8 - fp
        color = "red" if key == "wer" and delta > 0 else "green"
        table.add_row(
            f"{label} (mean)",
            f"{fp:.4f}",
            f"{int8:.4f}",
            f"[{color}]{delta:+.4f}[/{color}]",
        )
    console.print(table)
    console.print(f"quantization WER gap: [bold]{gap:+.4f}[/bold]")
    if problems:
        console.print("[red]breached gates:[/red]")
        for problem in problems:
            console.print(f"  [red]-[/red] {problem}")
    else:
        console.print("[green]all gates pass[/green]")


if __name__ == "__main__":
    raise SystemExit(main())
