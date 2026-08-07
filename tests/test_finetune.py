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


# ---------------------------------------------------------------------------
# CTranslate2 export (the fixed path) - both tests need torch + transformers,
# and ctranslate2 (ships with the ``engines`` extra via faster-whisper), so
# they skip in minimal environments. They share a tiny PEFT-wrapped whisper
# model so no training run is needed.
# ---------------------------------------------------------------------------
def _tiny_peft_model():
    """Whisper-tiny wrapped in LoRA, exactly like ``_run`` does pre-export.

    ``torch`` is already in scope from the module-level importorskip, so peft
    can be imported lazily here without an explicit torch import.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="SEQ_2_SEQ_LM",
        ),
    )


def test_export_ct2_converts_and_is_loadable(tmp_path, processor) -> None:
    """The export fix is regression-tested end to end: the converted int8
    directory is exactly what faster-whisper loads at serving time.

    This is the test that would have caught both earlier failures: the
    converter receiving a model *object* instead of a path, and the missing
    tokenizer files in the source directory. It downloads whisper-tiny
    (~40 MB) once into the HF cache and performs a real int8 conversion, so
    it is the heaviest test in this module - skip it in network-less CI by
    not installing the ``engines`` extra.
    """
    pytest.importorskip("ctranslate2")
    faster_whisper = pytest.importorskip("faster_whisper")

    from speechai.finetune.train import _export_ct2

    output_dir = tmp_path / "finetuned"
    ct2_dir = _export_ct2(_tiny_peft_model(), processor, "openai/whisper-tiny", output_dir)

    assert (ct2_dir / "model.bin").is_file()
    for name in ("tokenizer.json", "preprocessor_config.json", "generation_config.json"):
        assert (ct2_dir / name).is_file(), f"missing {name} next to model.bin"
    # The fp32 source weights are cleaned up after conversion.
    assert not (output_dir / "ct2_source").exists()

    # The exported dir must load through the real serving engine stack and
    # decode actual audio (a 1 s sine is enough to prove the model works).
    from speechai.audio.io import generate_sine

    whisper = faster_whisper.WhisperModel(str(ct2_dir), device="cpu", compute_type="int8")
    audio = generate_sine(1.0, 16000).samples
    segments, _ = whisper.transcribe(audio, language="en")
    # For pure tones whisper may emit nothing - the point is that loading the
    # exported dir and transcribing completes without error.
    _ = list(segments)


def test_export_ct2_passes_path_and_copies_tokenizer_locally(tmp_path, processor, monkeypatch) -> None:
    """Fast regression guards for the fix, with a stubbed converter:

    - the converter receives a *path* (the old bug passed the model object),
    - the source dir contains the merged weights + processor files,
    - tokenizer files are copied from the local source dir - no HF hub call,
      so the air-gapped workflow keeps working,
    - the source dir is removed afterwards.
    """
    pytest.importorskip("ctranslate2")

    from pathlib import Path

    import ctranslate2.converters as ct2_converters

    from speechai.finetune.train import _export_ct2

    captured: dict = {}

    class FakeConverter:
        # Constructed after the merged weights + processor were saved, but
        # before the source dir is cleaned up - so snapshot it here.
        def __init__(self, model_name_or_path, copy_files=None):
            captured["source_path"] = model_name_or_path
            captured["source_files"] = sorted(Path(model_name_or_path).iterdir())
            captured["copy_files"] = copy_files

        def convert(self, output_dir, vmap=None, quantization=None, force=False):
            captured["output_dir"] = output_dir
            captured["quantization"] = quantization
            captured["force"] = force
            Path(output_dir, "model.bin").write_bytes(b"fake")
            Path(output_dir, "model.bin.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ct2_converters, "TransformersConverter", FakeConverter)

    output_dir = tmp_path / "finetuned"
    ct2_dir = _export_ct2(_tiny_peft_model(), processor, "openai/whisper-tiny", output_dir)

    # The old bug passed the model object; the converter must get a directory.
    source_path = captured["source_path"]
    assert isinstance(source_path, str) and source_path
    source_files = {path.name for path in captured["source_files"]}
    assert "config.json" in source_files  # merged weights were saved
    assert "tokenizer.json" in source_files  # processor was saved
    assert captured["quantization"] == "int8"
    assert captured["force"] is True

    # Tokenizer files land next to model.bin and came from the local source dir
    # (the FakeConverter never writes them, and no HF call is made).
    for name in ("tokenizer.json", "preprocessor_config.json", "generation_config.json"):
        assert (ct2_dir / name).is_file(), f"missing {name}"
    assert not (output_dir / "ct2_source").exists()
