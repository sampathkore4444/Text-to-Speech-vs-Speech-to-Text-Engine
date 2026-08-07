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


def test_checkpoint_roundtrip(tmp_path) -> None:
    """A saved checkpoint restores model/optimizer/scheduler state and counters."""
    import torch

    from speechai.finetune.train import (
        TrainConfig,
        _load_checkpoint,
        _save_checkpoint,
    )

    def make_state(seed: int):
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        # One optimizer step so AdamW has non-empty state to serialize.
        model.zero_grad()
        loss = model(torch.randn(4, 4)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return model, optimizer, scheduler

    config = TrainConfig(data="x", base_model="openai/whisper-tiny")
    path = tmp_path / "checkpoints" / "latest.pt"

    model_1, optimizer_1, scheduler_1 = make_state(seed=7)
    _save_checkpoint(
        path, config=config, model=model_1, optimizer=optimizer_1,
        scheduler=scheduler_1, next_epoch=2, global_step=5, best_val_wer=0.123,
    )

    model_2, optimizer_2, scheduler_2 = make_state(seed=99)
    next_epoch, global_step, best_val_wer = _load_checkpoint(
        path, config=config, model=model_2, optimizer=optimizer_2,
        scheduler=scheduler_2, device="cpu",
    )
    assert (next_epoch, global_step, best_val_wer) == (2, 5, 0.123)
    for p1, p2 in zip(model_1.parameters(), model_2.parameters(), strict=True):
        assert torch.equal(p1.detach(), p2.detach())
    # Optimizer + scheduler state survived the round trip.
    assert len(optimizer_2.state) == len(optimizer_1.state)
    assert optimizer_2.param_groups[0]["lr"] == optimizer_1.param_groups[0]["lr"]
    assert scheduler_2.get_last_lr() == scheduler_1.get_last_lr()


def test_resume_rejects_config_mismatch(tmp_path) -> None:
    """Resuming with a different model-defining config must fail loudly."""
    import torch

    from speechai.finetune.train import (
        TrainConfig,
        _load_checkpoint,
        _save_checkpoint,
    )

    config = TrainConfig(data="x", base_model="openai/whisper-tiny", lora_r=8)
    path = tmp_path / "ckpt.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    _save_checkpoint(
        path, config=config, model=model, optimizer=optimizer,
        scheduler=scheduler, next_epoch=1, global_step=0,
    )

    other = TrainConfig(data="x", base_model="openai/whisper-base", lora_r=8)
    with pytest.raises(ValueError, match="base_model"):
        _load_checkpoint(
            path, config=other, model=model, optimizer=optimizer,
            scheduler=scheduler, device="cpu",
        )


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
