"""Audio IO tests: load, resample, encode/decode."""

from __future__ import annotations

import numpy as np
import pytest

from speechai.audio.io import (
    AudioBuffer,
    generate_sine,
    load_audio,
    pcm16_bytes,
    to_asr_audio,
    write_wav,
)
from speechai.core.errors import AudioFormatError


def test_duration() -> None:
    audio = generate_sine(0.5, 16000)
    assert audio.duration_seconds == pytest.approx(0.5, abs=1e-3)


def test_write_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "tone.wav"
    write_wav(path, generate_sine(0.4, 22050))
    loaded = load_audio(path)
    assert loaded.sample_rate == 22050
    assert loaded.duration_seconds == pytest.approx(0.4, abs=0.02)


def test_resample_44100_to_16000() -> None:
    audio = generate_sine(0.1, 44100)
    resampled = audio.resample(16000)
    assert resampled.sample_rate == 16000
    assert resampled.samples.shape[0] == pytest.approx(1600, rel=0.05)


def test_to_asr_audio_standardizes() -> None:
    audio = generate_sine(0.2, 48000)
    asr = to_asr_audio(audio)
    assert asr.sample_rate == 16000


def test_wav_bytes_roundtrip() -> None:
    audio = generate_sine(0.3, 16000)
    wav = audio.to_wav_bytes()
    assert wav[:4] == b"RIFF"
    decoded = AudioBuffer.from_wav_bytes(wav)
    assert decoded.sample_rate == 16000
    assert decoded.duration_seconds == pytest.approx(0.3, abs=0.02)


def test_pcm16_roundtrip() -> None:
    audio = generate_sine(0.2, 16000)
    pcm = pcm16_bytes(audio.samples)
    decoded = AudioBuffer.from_pcm16(pcm, 16000)
    assert np.max(np.abs(decoded.samples - audio.samples)) < 1e-3


def test_load_invalid_audio_raises() -> None:
    with pytest.raises(AudioFormatError):
        load_audio(b"this is not audio")
