"""Evaluation runner: run the STT engine over a dataset and report metrics."""

from __future__ import annotations

import logging
from pathlib import Path

from speechai.audio.io import load_audio, to_asr_audio
from speechai.core import metrics
from speechai.core.timing import Stopwatch, compute_rtf
from speechai.eval.loader import EvalExample
from speechai.eval.metrics import (
    EvaluationReport,
    UtteranceResult,
    char_error_rate,
    word_error_rate,
)
from speechai.stt.base import STTEngine, STTOptions

logger = logging.getLogger(__name__)


def run_evaluation(
    engine: STTEngine,
    examples: list[EvalExample],
    *,
    language: str | None = None,
    dataset_name: str = "default",
) -> EvaluationReport:
    """Transcribe every example and build an aggregate report.

    This is the reference implementation of the quality/SLA gates the job
    description mentions (WER, CER, RTF, latency): the resulting report can be
    compared against ``eval.default_wer_tolerance`` / ``eval.default_rtf_tolerance``
    in CI, and the aggregates are exported to Prometheus gauges.
    """
    engine.load()
    results: list[UtteranceResult] = []
    for index, example in enumerate(examples, start=1):
        audio = to_asr_audio(load_audio(example.audio))
        stopwatch = Stopwatch()
        result = engine.transcribe(audio, STTOptions(language=language))
        latency = stopwatch.elapsed()
        rtf = compute_rtf(latency, result.duration_seconds)
        results.append(
            UtteranceResult(
                audio=str(example.audio),
                reference=example.reference,
                hypothesis=result.text,
                wer=word_error_rate(example.reference, result.text),
                cer=char_error_rate(example.reference, result.text),
                audio_duration=result.duration_seconds,
                latency_seconds=latency,
                rtf=rtf,
                confidence=result.avg_confidence,
            )
        )
        logger.info(
            "evaluated utterance",
            extra={
                "index": index,
                "total": len(examples),
                "wer": results[-1].wer,
                "rtf": round(rtf, 3),
            },
        )

    report = EvaluationReport(engine=engine.name, dataset=dataset_name, results=results)
    _push_gauges(report)
    return report


def _push_gauges(report: EvaluationReport) -> None:
    """Export aggregate metrics to Prometheus for alerting (e.g. regression gates)."""
    agg = report.aggregates
    metrics.stt_wer.labels(report.dataset).set(agg["wer"]["mean"])
    metrics.stt_cer.labels(report.dataset).set(agg["cer"]["mean"])
    metrics.stt_rtf_mean.labels(report.dataset).set(agg["rtf"]["mean"])


def assert_within_tolerance(
    report: EvaluationReport,
    *,
    wer_tolerance: float = 0.10,
    rtf_tolerance: float = 0.50,
) -> None:
    """Raise if aggregate quality metrics breach the regression gates."""
    agg = report.aggregates
    problems: list[str] = []
    if agg["wer"]["mean"] > wer_tolerance:
        problems.append(f"mean WER {agg['wer']['mean']:.3f} > tolerance {wer_tolerance}")
    if agg["rtf"]["mean"] > rtf_tolerance:
        problems.append(f"mean RTF {agg['rtf']['mean']:.3f} > tolerance {rtf_tolerance}")
    if problems:
        raise ValueError("Evaluation regression gate failed: " + "; ".join(problems))


def run_from_manifest(
    engine: STTEngine,
    manifest: str | Path,
    *,
    language: str | None = None,
) -> EvaluationReport:
    from speechai.eval.loader import load_manifest

    examples = load_manifest(manifest)
    dataset_name = Path(manifest).stem
    return run_evaluation(engine, examples, language=language, dataset_name=dataset_name)
