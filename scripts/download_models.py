"""Download model artifacts for on-premise inference.

- Whisper (faster-whisper) models download automatically from the Hugging Face
  Hub on first load; this script optionally warms that cache.
- Piper voices must be downloaded explicitly into ``data/models/voices/``.

Usage:
    python scripts/download_models.py                       # Piper default voice
    python scripts/download_models.py --piper-voice en_US-amy-medium
    python scripts/download_models.py --whisper-size base   # also warm the Whisper cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

PIPER_HF_ROOT = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

PIPER_VOICES: dict[str, tuple[str, list[str]]] = {
    # voice -> (subpath prefix, file suffixes)
    "en_US-lessac-medium": ("en/en_US/lessac/medium/en_US-lessac-medium", [".onnx", ".onnx.json"]),
    "en_US-amy-medium": ("en/en_US/amy/medium/en_US-amy-medium", [".onnx", ".onnx.json"]),
    "en_GB-alan-medium": ("en/en_GB/alan/medium/en_GB-alan-medium", [".onnx", ".onnx.json"]),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already present: {dest}")
        return
    print(f"  downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "bank-speech-ai"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_piper(voice: str, voices_dir: Path) -> None:
    if voice not in PIPER_VOICES:
        print(f"Unknown voice {voice!r}; available: {', '.join(sorted(PIPER_VOICES))}")
        sys.exit(2)
    prefix, suffixes = PIPER_VOICES[voice]
    for suffix in suffixes:
        _download(f"{PIPER_HF_ROOT}/{prefix}{suffix}", voices_dir / f"{voice}{suffix}")
    print(f"Piper voice {voice} ready in {voices_dir}")


def warm_whisper(model_size: str) -> None:
    print(f"Warming faster-whisper cache for model {model_size!r} (first download may take a while)...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper not installed - skipping. Install with: pip install -e '.[engines]'")
        return
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    del model
    print("Whisper model ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piper-voice", default="en_US-lessac-medium")
    parser.add_argument("--whisper-size", default=None, help="e.g. base - warms the Whisper cache")
    parser.add_argument("--models-dir", default="data/models", help="Model root directory")
    args = parser.parse_args()

    voices_dir = Path(args.models_dir) / "voices"
    if args.piper_voice:
        download_piper(args.piper_voice, voices_dir)
    if args.whisper_size:
        warm_whisper(args.whisper_size)
    print("Done.")


if __name__ == "__main__":
    main()
