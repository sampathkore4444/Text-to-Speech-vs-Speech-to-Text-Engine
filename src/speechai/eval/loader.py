"""Evaluation dataset loading: JSONL, CSV or a directory of pairs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalExample:
    audio: Path
    reference: str


def load_manifest(path: str | Path) -> list[EvalExample]:
    """Load an evaluation manifest.

    Supported formats:
    - JSONL: one object per line with ``audio`` and ``reference`` keys
    - CSV: columns ``audio``, ``reference``
    - Directory: pairs of ``<name>.wav`` + ``<name>.txt``
    """
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if manifest.is_dir():
        return load_directory(manifest)
    suffix = manifest.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(manifest)
    if suffix == ".csv":
        return _load_csv(manifest)
    raise ValueError(f"Unsupported manifest format: {suffix} (use .jsonl, .csv or a directory)")


def load_directory(directory: str | Path) -> list[EvalExample]:
    """Discover ``<stem>.wav`` + ``<stem>.txt`` pairs inside a directory."""
    root = Path(directory)
    examples: list[EvalExample] = []
    for audio in sorted(root.glob("*.wav")):
        text = audio.with_suffix(".txt")
        if text.is_file():
            examples.append(EvalExample(audio=audio, reference=text.read_text(encoding="utf-8").strip()))
    return examples


def _load_jsonl(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        examples.append(EvalExample(audio=_resolve(record["audio"]), reference=record.get("reference", "")))
    return examples


def _load_csv(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            examples.append(EvalExample(audio=_resolve(row["audio"]), reference=row.get("reference", "")))
    return examples


def _resolve(audio: str) -> Path:
    path = Path(audio)
    return path if path.is_absolute() else path
