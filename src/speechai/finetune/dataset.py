"""PyTorch dataset for Whisper LoRA fine-tuning built on platform manifests.

Reuses ``speechai.eval.loader`` so any platform manifest (JSONL / CSV / a
directory of ``wav + txt`` pairs) can be used directly for training. Audio is
decoded with the platform's own ``speechai.audio.io`` (soundfile) and converted
to Whisper log-mel features with the HF feature extractor - no ffmpeg needed.

The dataset is built eagerly in memory, which suits fine-tuning corpora of up
to a few thousand utterances. For larger corpora, precompute features to disk
and memory-map them (see ``docs/finetuning.md``).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from speechai.audio.io import load_audio, to_asr_audio
from speechai.eval.loader import EvalExample, load_manifest

logger = logging.getLogger(__name__)

LABEL_MAX_LENGTH = 448  # Whisper token limit per utterance
MEL_FRAMES = 3000  # Whisper's fixed 30 s log-mel window (transformers >= 5 enforces it)
WHISPER_MAX_SAMPLES = 480000  # 30 s @ 16 kHz


class WhisperDataset:
    """Mel-feature + label dataset for one manifest."""

    def __init__(
        self,
        examples: list[EvalExample],
        processor,
        *,
        max_audio_seconds: float = 30.0,
        sample_rate: int = 16000,
    ) -> None:
        # torch is imported lazily so this module stays importable without the
        # heavy finetune stack (useful for docs/CLI error paths).
        import torch  # noqa: F401

        self.examples = list(examples)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        # Whisper consumes fixed 30 s windows; shorter audio is padded, longer
        # audio must be truncated.
        self._max_samples = min(int(max_audio_seconds * sample_rate), WHISPER_MAX_SAMPLES)
        self._pad_value = float(getattr(processor.feature_extractor, "padding_value", 0.0))
        self._items = [self._build(example) for example in self.examples]
        self.total_audio_seconds = sum(item["duration_seconds"] for item in self._items)
        logger.info(
            "built dataset",
            extra={
                "utterances": len(self._items),
                "audio_seconds": round(self.total_audio_seconds, 1),
            },
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict:
        return self._items[index]

    @property
    def items(self) -> list[dict]:
        """Raw items (features, labels, metadata) - used by the WER evaluator."""
        return self._items

    # ------------------------------------------------------------------
    def _build(self, example: EvalExample) -> dict:
        audio = to_asr_audio(load_audio(example.audio))
        samples = audio.samples[: self._max_samples]
        features = self.processor(
            audio=samples, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features[0]
        features = _pad_or_truncate(features, MEL_FRAMES, self._pad_value)
        labels = self.processor.tokenizer(
            example.reference, return_tensors="pt", max_length=LABEL_MAX_LENGTH, truncation=True
        ).input_ids[0]
        return {
            "input_features": features,  # (80, 3000) log-mel
            "labels": labels,  # (L,) token ids
            "reference": example.reference,
            "audio": str(example.audio),
            "duration_seconds": float(samples.size) / self.sample_rate,
        }


def collate_whisper_batch(batch: list[dict]) -> dict:
    """Pad features and labels to the longest item in the batch.

    Label padding uses -100 (the ignore index), which is how HF Whisper masks
    loss over padding.
    """
    import torch  # lazy

    if not batch:
        return {}
    features = [item["input_features"] for item in batch]
    max_frames = max(f.size(-1) for f in features)
    input_features = torch.stack(
        [torch.nn.functional.pad(f, (0, max_frames - f.size(-1))) for f in features]
    )
    labels = [item["labels"] for item in batch]
    max_tokens = max(label.size(0) for label in labels)
    label_ids = torch.full((len(labels), max_tokens), -100, dtype=torch.long)
    for index, tokens in enumerate(labels):
        label_ids[index, : tokens.size(0)] = tokens
    return {
        "input_features": input_features,
        "labels": label_ids,
        "references": [item["reference"] for item in batch],
        "audio": [item["audio"] for item in batch],
        "durations": [item["duration_seconds"] for item in batch],
    }


def _pad_or_truncate(features, target: int, pad_value: float):
    """Pad (right) or truncate mel features to Whisper's fixed 30 s window."""
    import torch

    n_frames = features.size(-1)
    if n_frames == target:
        return features
    if n_frames < target:
        return torch.nn.functional.pad(features, (0, target - n_frames), value=pad_value)
    return features[..., :target]


def split_manifest(
    manifest: str | Path, *, val_split: float = 0.1, seed: int = 42
) -> tuple[list[EvalExample], list[EvalExample]]:
    """Deterministic train/validation split of a manifest."""
    examples = load_manifest(manifest)
    if not examples:
        raise ValueError(f"No examples found in manifest: {manifest}")
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_split)) if val_split > 0 else 0
    return examples[n_val:], examples[:n_val]
