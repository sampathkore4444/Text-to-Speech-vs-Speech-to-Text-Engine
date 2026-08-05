"""Generate bank-domain sample utterances with the Piper TTS engine.

Produces:
- ``data/samples/<name>.wav``  (16 kHz mono, STT-ready)
- ``data/samples/<name>.txt``  (reference transcripts)
- ``data/manifest.jsonl``      (evaluation manifest referencing the samples)

This gives you an instant STT demo loop without recording anything:
    python scripts/make_sample_audio.py
    speechai transcribe data/samples/sample_01_account_balance.wav --json
    speechai evaluate data/manifest.jsonl
"""

from __future__ import annotations

import json
import sys

from speechai.audio.io import write_wav
from speechai.core.config import Settings
from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.tts.base import build_tts_engine
from speechai.tts.textnorm import TextNormalizer

PHRASES: list[tuple[str, str]] = [
    (
        "sample_01_account_balance",
        "Your account balance is one thousand two hundred and fifty dollars and fifty cents as of today.",
    ),
    (
        "sample_02_card_verification",
        "Please verify the last four digits of your card to continue.",
    ),
    (
        "sample_03_interest_rate",
        "The interest rate on your fixed deposit is three point five percent per annum.",
    ),
    (
        "sample_04_fraud_alert",
        "We detected unusual activity on your account. Press one to confirm a recent transaction, or two to speak with an agent.",
    ),
    (
        "sample_05_appointment",
        "Your branch appointment is scheduled for the fifteenth of August at ten thirty in the morning.",
    ),
]


def main() -> int:
    settings = Settings.load()
    settings.ensure_dirs()
    redactor = Redactor(RedactionPolicy.from_settings(settings.redaction))
    normalizer = TextNormalizer(redactor)

    print("Loading TTS engine...")
    engine = build_tts_engine(settings)
    engine.load()

    samples_dir = settings.data_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.data_dir / "manifest.jsonl"

    manifest_lines: list[str] = []
    for name, phrase in PHRASES:
        normalized = normalizer.normalize(phrase)
        result = engine.synthesize(normalized.text)
        audio_16k = result.audio.resample(16000)
        wav_path = samples_dir / f"{name}.wav"
        txt_path = samples_dir / f"{name}.txt"
        write_wav(wav_path, audio_16k)
        txt_path.write_text(phrase, encoding="utf-8")
        manifest_lines.append(
            json.dumps({"audio": str(wav_path.resolve()), "reference": phrase})
        )
        print(f"  wrote {wav_path} ({audio_16k.duration_seconds:.2f}s, redacted={normalized.redacted})")

    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(manifest_lines)} samples to {samples_dir}")
    print(f"Evaluation manifest: {manifest_path}")
    print("\nNext steps:")
    print("  speechai transcribe data/samples/sample_01_account_balance.wav --json")
    print("  speechai evaluate data/manifest.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
