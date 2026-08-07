"""Command-line interface: ``speechai transcribe | synthesize | evaluate | models``.

Examples::

    speechai transcribe call.wav --json
    speechai synthesize "Your balance is $1,250.50" -o balance.wav
    speechai evaluate data/manifest.jsonl --gate
    speechai models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from speechai.audio.io import load_audio, to_asr_audio, write_wav
from speechai.core.config import Settings
from speechai.core.logging import setup_logging
from speechai.core.tracking import ExperimentTracker
from speechai.eval.runner import assert_within_tolerance, run_from_manifest
from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.stt.base import STTOptions, build_stt_engine
from speechai.stt.postprocess import TextPostProcessor
from speechai.tts.base import build_tts_engine
from speechai.tts.textnorm import TextNormalizer

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speechai",
        description="Bank Speech AI platform CLI (STT + TTS + evaluation).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe an audio file")
    p_transcribe.add_argument("audio", help="Path to audio file (wav/mp3/flac/...)")
    p_transcribe.add_argument("--language", default=None, help="Override language (e.g. en)")
    p_transcribe.add_argument("--no-redact", action="store_true", help="Disable PII redaction")
    p_transcribe.add_argument(
        "--vad",
        dest="vad_filter",
        action="store_true",
        default=None,
        help="Enable the voice-activity filter (clean speech only; may drop noisy audio)",
    )
    p_transcribe.add_argument(
        "--no-vad",
        dest="vad_filter",
        action="store_false",
        help="Disable the voice-activity filter (transcribe everything; default)",
    )
    p_transcribe.add_argument("--json", action="store_true", help="Emit raw JSON")

    p_synthesize = sub.add_parser("synthesize", help="Synthesize speech from text")
    p_synthesize.add_argument("text", help="Text to synthesize")
    p_synthesize.add_argument("-o", "--output", default="output.wav", help="Output WAV path")
    p_synthesize.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.5-2.0)")

    p_evaluate = sub.add_parser("evaluate", help="Evaluate STT quality on a manifest")
    p_evaluate.add_argument("manifest", help="JSONL/CSV manifest or directory of wav+txt pairs")
    p_evaluate.add_argument("--language", default=None)
    p_evaluate.add_argument("--report", default=None, help="Where to write report.json")
    p_evaluate.add_argument("--gate", action="store_true", help="Fail if WER/RTF tolerances breached")
    p_evaluate.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow experiment tracking"
    )

    sub.add_parser("models", help="Show engine/model configuration")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    setup_logging(
        settings.service.log_level, settings.service.log_format, settings.service.name
    )
    redactor = Redactor(RedactionPolicy.from_settings(settings.redaction))
    try:
        if args.command == "transcribe":
            return _transcribe(args, settings, redactor)
        if args.command == "synthesize":
            return _synthesize(args, settings, redactor)
        if args.command == "evaluate":
            return _evaluate(args, settings)
        if args.command == "models":
            return _models(settings)
    except Exception as exc:  # surface clean errors
        console.print(f"[red]error:[/red] {exc}")
        return 1
    return 1


# ---------------------------------------------------------------------------
def _transcribe(args: argparse.Namespace, settings: Settings, redactor: Redactor) -> int:
    engine = build_stt_engine(settings)
    audio = to_asr_audio(load_audio(args.audio))
    result = engine.transcribe(
        audio, STTOptions(language=args.language, vad_filter=args.vad_filter)
    )
    processed = TextPostProcessor(redactor).process(result.text, redact=not args.no_redact)

    if args.json:
        payload = {
            "text": processed.text,
            "language": result.language,
            "engine": result.engine,
            "redacted": processed.redacted,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end, "confidence": s.confidence}
                for s in result.segments
            ],
            "metrics": {
                "audio_duration_seconds": result.duration_seconds,
                "latency_seconds": result.latency_seconds,
                "rtf": result.rtf,
                "confidence": result.avg_confidence,
            },
        }
        console.print(json.dumps(payload, indent=2))
        return 0

    table = Table(title="Transcription", show_lines=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_row("Text", processed.text)
    table.add_row("Language", result.language or "auto")
    table.add_row("Engine", result.engine)
    table.add_row("Redacted", "yes" if processed.redacted else "no")
    table.add_row("Audio duration", f"{result.duration_seconds:.2f}s")
    table.add_row("Latency", f"{result.latency_seconds:.3f}s")
    table.add_row("RTF", f"{result.rtf:.3f}")
    table.add_row("Confidence", f"{result.avg_confidence if result.avg_confidence is not None else '-'}")
    console.print(table)
    return 0


def _synthesize(args: argparse.Namespace, settings: Settings, redactor: Redactor) -> int:
    engine = build_tts_engine(settings)
    normalized = TextNormalizer(redactor).normalize(args.text)
    result = engine.synthesize(normalized.text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output, result.audio)
    console.print(
        f"[green]wrote[/green] {output}  "
        f"({result.audio.duration_seconds:.2f}s audio, {result.latency_seconds:.3f}s latency, "
        f"RTF {result.rtf:.3f}, redacted={normalized.redacted})"
    )
    return 0


def _evaluate(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_stt_engine(settings)
    report = run_from_manifest(engine, args.manifest, language=args.language)
    console.print(report.to_text())
    report_path = args.report or str(
        settings.eval_report_dir / f"{report.dataset}-{report.engine}.json"
    )
    report.export_json(report_path)
    console.print(f"[green]report written to[/green] {report_path}")
    tracker = _track_evaluation(settings, report, report_path, enabled=not args.no_mlflow)
    if args.gate:
        try:
            assert_within_tolerance(
                report,
                wer_tolerance=settings.eval.default_wer_tolerance,
                rtf_tolerance=settings.eval.default_rtf_tolerance,
            )
            console.print("[green]regression gates passed[/green]")
        except ValueError:
            # A breached quality gate is a failed run in the tracking UI.
            if tracker is not None:
                tracker.end(status="FAILED")
            raise
    if tracker is not None:
        tracker.end()
    return 0


def _track_evaluation(
    settings: Settings, report, report_path: str | Path, *, enabled: bool
) -> ExperimentTracker | None:
    """Record an evaluation run to MLflow (no-op unless configured).

    Opt-in via ``tracking.enabled`` in config, ``MLFLOW_TRACKING_URI``, or the
    ``--mlflow-tracking-uri`` style URI; ``--no-mlflow`` overrides it off.
    Starts the run and logs params/metrics/artifact but does **not** end it -
    the caller ends it so the status reflects the ``--gate`` outcome.
    Returns the active tracker, or ``None`` when tracking is disabled.
    """
    tracking_on = enabled and (settings.tracking.enabled or bool(os.environ.get("MLFLOW_TRACKING_URI")))
    tracker = ExperimentTracker(
        enabled=tracking_on,
        tracking_uri=settings.tracking.tracking_uri or None,
        experiment_name=settings.tracking.experiment_name,
        run_name=f"{report.dataset}-{report.engine}",
    )
    if not tracker.enabled:
        return None
    tracker.start(tags={"tool": "speechai-evaluate", "dataset": report.dataset, "engine": report.engine})
    tracker.log_params(
        {
            "dataset": report.dataset,
            "engine": report.engine,
            "model": settings.stt.model_path or settings.stt.model_size,
            "language": settings.stt.language,
            "wer_tolerance": settings.eval.default_wer_tolerance,
            "rtf_tolerance": settings.eval.default_rtf_tolerance,
            "n_utterances": len(report.results),
        }
    )
    agg = report.aggregates
    metrics_: dict[str, float] = {}
    for key in ("wer", "cer", "rtf", "latency_seconds"):
        for stat in ("mean", "median", "p90"):
            metrics_[f"{key}_{stat}"] = agg[key][stat]
    tracker.log_metrics(metrics_)
    tracker.log_artifact(report_path)
    return tracker


def _models(settings: Settings) -> int:
    table = Table(title="Models")
    table.add_column("Engine", style="bold cyan")
    table.add_column("Config")
    table.add_row("STT", f"{settings.stt.engine} / {settings.stt.model_size} / {settings.stt.device} / {settings.stt.compute_type}")
    table.add_row("STT model path", settings.stt.model_path or "-")
    table.add_row("TTS", f"{settings.tts.engine} / {settings.tts.voice}")
    table.add_row("VAD", settings.vad.backend)
    table.add_row("Redaction", f"{settings.redaction.mode} (enabled={settings.redaction.enabled})")
    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
