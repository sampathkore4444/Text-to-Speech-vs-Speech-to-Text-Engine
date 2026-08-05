"""Evaluation metrics tests: WER, CER, RTF and report aggregates."""

from __future__ import annotations

from speechai.core.timing import compute_rtf
from speechai.eval.metrics import (
    EvaluationReport,
    UtteranceResult,
    char_error_rate,
    word_error_rate,
)


def test_wer_perfect() -> None:
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_half() -> None:
    assert word_error_rate("hello world", "hello") == 0.5


def test_wer_empty_hypothesis() -> None:
    assert word_error_rate("hello world", "") == 1.0


def test_cer() -> None:
    assert char_error_rate("hello", "hello") == 0.0
    assert char_error_rate("hello", "hell") > 0.0


def test_rtf() -> None:
    assert compute_rtf(0.5, 1.0) == 0.5
    assert compute_rtf(1.0, 0.0) == 0.0


def test_report_aggregates() -> None:
    report = EvaluationReport(
        engine="fake",
        dataset="test",
        results=[
            UtteranceResult("a.wav", "hello world", "hello world", 0.0, 0.0, 1.0, 0.5, 0.5, 0.9),
            UtteranceResult("b.wav", "goodbye moon", "goodbye", 0.5, 0.2, 2.0, 1.0, 0.5, 0.8),
        ],
    )
    agg = report.aggregates
    assert agg["wer"]["mean"] == 0.25
    assert agg["wer"]["median"] == 0.25
    assert report.to_dict()["n_utterances"] == 2
    text = report.to_text()
    assert "WER" in text


def test_report_export_json(tmp_path) -> None:
    report = EvaluationReport(engine="fake", dataset="test", results=[])
    path = report.export_json(tmp_path / "report.json")
    assert path.is_file()
    assert '"engine": "fake"' in path.read_text(encoding="utf-8")
