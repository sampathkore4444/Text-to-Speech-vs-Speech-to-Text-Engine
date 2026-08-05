"""Evaluation package: metrics, dataset loading and the evaluation runner."""

from speechai.eval.loader import EvalExample, load_directory, load_manifest
from speechai.eval.metrics import (
    EvaluationReport,
    UtteranceResult,
    char_error_rate,
    word_error_rate,
)
from speechai.eval.runner import assert_within_tolerance, run_evaluation, run_from_manifest

__all__ = [
    "EvalExample",
    "EvaluationReport",
    "UtteranceResult",
    "assert_within_tolerance",
    "char_error_rate",
    "load_directory",
    "load_manifest",
    "run_evaluation",
    "run_from_manifest",
    "word_error_rate",
]
