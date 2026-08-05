"""Evaluation metrics and reporting: WER, CER, RTF, latency."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jiwer


@dataclass
class UtteranceResult:
    """Per-utterance evaluation outcome."""

    audio: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    audio_duration: float
    latency_seconds: float
    rtf: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "audio_duration": round(self.audio_duration, 3),
            "latency_seconds": round(self.latency_seconds, 3),
            "rtf": round(self.rtf, 3),
            "confidence": self.confidence,
        }


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate via jiwer (0.0 = perfect)."""
    try:
        return float(jiwer.wer(reference.strip(), hypothesis.strip()))
    except Exception:  # pragma: no cover - jiwer edge cases
        return 1.0


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate via jiwer."""
    try:
        return float(jiwer.cer(reference.strip(), hypothesis.strip()))
    except Exception:  # pragma: no cover - jiwer edge cases
        return 1.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p90": round(_percentile(values, 90), 4),
    }


@dataclass
class EvaluationReport:
    """Aggregate results of an evaluation run."""

    engine: str
    dataset: str
    results: list[UtteranceResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def aggregates(self) -> dict[str, dict[str, float]]:
        return {
            "wer": _aggregate([r.wer for r in self.results]),
            "cer": _aggregate([r.cer for r in self.results]),
            "rtf": _aggregate([r.rtf for r in self.results]),
            "latency_seconds": _aggregate([r.latency_seconds for r in self.results]),
        }

    @property
    def total_audio_seconds(self) -> float:
        return round(sum(r.audio_duration for r in self.results), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "n_utterances": len(self.results),
            "total_audio_seconds": self.total_audio_seconds,
            "aggregates": self.aggregates,
            "utterances": [r.to_dict() for r in self.results],
        }

    def to_text(self) -> str:
        agg = self.aggregates
        lines = [
            f"Evaluation report  engine={self.engine}  dataset={self.dataset}",
            f"Utterances: {len(self.results)}   total audio: {self.total_audio_seconds}s",
            "",
            "Metric               mean     median   p90",
            "-------------------- -------- -------- --------",
        ]
        for key, label in (("wer", "WER"), ("cer", "CER"), ("rtf", "RTF"), ("latency_seconds", "Latency (s)")):
            row = agg[key]
            lines.append(f"{label:<20} {row['mean']:<8.4f} {row['median']:<8.4f} {row['p90']:<8.4f}")
        lines.append("")
        lines.append("Worst 5 utterances by WER:")
        for item in sorted(self.results, key=lambda r: r.wer, reverse=True)[:5]:
            lines.append(f"  wer={item.wer:.2f}  {item.audio}: {item.reference!r} -> {item.hypothesis!r}")
        return "\n".join(lines)

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target
