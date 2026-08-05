"""Audio loading, resampling and encoding utilities.

All internal audio is represented as ``AudioBuffer``: mono ``float32`` numpy
samples plus a sample rate. ASR consumes mono audio at 16 kHz.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from speechai.core.errors import AudioFormatError

ASR_SAMPLE_RATE = 16000
_AudioSource = str | Path | bytes


@dataclass
class AudioBuffer:
    """Mono float32 audio samples with a sample rate."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        self.samples = np.ascontiguousarray(arr, dtype=np.float32)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0 or self.samples.size == 0:
            return 0.0
        return float(self.samples.shape[0]) / float(self.sample_rate)

    def resample(self, target_rate: int) -> AudioBuffer:
        return AudioBuffer(resample(self.samples, self.sample_rate, target_rate), target_rate)

    def to_pcm16(self) -> bytes:
        return pcm16_bytes(self.samples)

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, self.samples, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @classmethod
    def from_wav_bytes(cls, data: bytes) -> AudioBuffer:
        try:
            samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except Exception as exc:  # soundfile raises RuntimeError/LibFormatError
            raise AudioFormatError(f"Could not decode WAV data: {exc}") from exc
        return cls(np.asarray(samples, dtype=np.float32), int(rate))

    @classmethod
    def from_pcm16(cls, data: bytes, sample_rate: int) -> AudioBuffer:
        if not data:
            return cls(np.empty(0, dtype=np.float32), sample_rate)
        raw = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return cls(raw, sample_rate)


def load_audio(source: _AudioSource) -> AudioBuffer:
    """Decode any soundfile-supported format (wav/flac/mp3/ogg/...) to AudioBuffer."""
    try:
        if isinstance(source, (str, Path)):
            samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
        else:
            samples, rate = sf.read(io.BytesIO(source), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioFormatError(f"Could not decode audio: {exc}") from exc
    if samples.size == 0:
        raise AudioFormatError("Audio contains no samples")
    return AudioBuffer(np.asarray(samples, dtype=np.float32), int(rate))


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 1-D audio. Uses soxr when available, linear fallback otherwise."""
    if src_rate == dst_rate:
        return np.asarray(samples, dtype=np.float32)
    try:
        import soxr  # optional high-quality resampler (engines extra)

        return np.asarray(soxr.resample(samples, src_rate, dst_rate), dtype=np.float32)
    except ImportError:
        return _resample_linear(samples, src_rate, dst_rate)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear-interpolation resampler (documented trade-off: aliasing)."""
    ratio = dst_rate / src_rate
    n_out = max(1, int(round(samples.shape[0] * ratio)))
    indices = np.arange(n_out, dtype=np.float64) / ratio
    x0 = indices.astype(np.int64)
    x1 = np.minimum(x0 + 1, samples.shape[0] - 1)
    frac = indices - x0
    return (samples[x0] * (1.0 - frac) + samples[x1] * frac).astype(np.float32)


def to_asr_audio(audio: AudioBuffer) -> AudioBuffer:
    """Convert any audio to the ASR standard: mono float32 @ 16 kHz."""
    return audio.resample(ASR_SAMPLE_RATE)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] samples to little-endian 16-bit PCM bytes."""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def write_wav(path: str | Path, audio: AudioBuffer) -> None:
    """Write an AudioBuffer to disk as a 16-bit WAV file."""
    sf.write(str(path), audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")


def generate_sine(duration_seconds: float, sample_rate: int, freq: float = 440.0) -> AudioBuffer:
    """Synthetic tone - useful for tests and VAD checks."""
    n = int(duration_seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    samples = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    return AudioBuffer(samples, sample_rate)
