"""LoRA fine-tuning support tests.

Skipped automatically unless the ``finetune`` extra (torch + transformers) is
installed. The processor fixture downloads the tiny Whisper tokenizer config
once from the Hugging Face Hub.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from speechai.audio.io import generate_sine, write_wav  # noqa: E402
from speechai.eval.loader import EvalExample  # noqa: E402
from speechai.finetune.dataset import (  # noqa: E402
    WhisperDataset,
    collate_whisper_batch,
    split_manifest,
)


@pytest.fixture(scope="module")
def processor():
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained("openai/whisper-tiny")


@pytest.fixture
def sample_examples(tmp_path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for index, duration in enumerate((0.4, 0.6)):
        path = tmp_path / f"sample_{index}.wav"
        write_wav(path, generate_sine(duration, 16000))
        examples.append(EvalExample(path, f"hello world number {index}"))
    return examples


def test_dataset_builds(processor, sample_examples) -> None:
    dataset = WhisperDataset(sample_examples, processor)
    assert len(dataset) == 2
    item = dataset[0]
    assert item["input_features"].dim() == 2
    assert item["input_features"].size(0) == 80  # log-mel bins
    assert item["input_features"].size(1) > 0
    assert item["labels"].size(0) > 0
    assert dataset.total_audio_seconds > 0


def test_collate_pads_to_longest(processor, sample_examples) -> None:
    dataset = WhisperDataset(sample_examples, processor)
    batch = collate_whisper_batch([dataset[0], dataset[1]])
    assert batch["input_features"].shape[0] == 2
    expected_frames = max(
        dataset[0]["input_features"].size(1), dataset[1]["input_features"].size(1)
    )
    assert batch["input_features"].shape[2] == expected_frames
    assert (batch["labels"] == -100).any() or batch["labels"].size(1) > 0


def test_collate_empty_batch() -> None:
    assert collate_whisper_batch([]) == {}


def test_split_manifest(tmp_path, sample_examples) -> None:
    manifest = tmp_path / "manifest.jsonl"
    lines = [
        json.dumps({"audio": str(example.audio), "reference": example.reference})
        for example in sample_examples
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")
    train, val = split_manifest(manifest, val_split=0.5, seed=7)
    assert len(train) + len(val) == 2
    assert len(val) == 1
