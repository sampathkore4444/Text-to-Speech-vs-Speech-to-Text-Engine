"""VAD and streaming utterance segmentation tests."""

from __future__ import annotations

import numpy as np
import pytest

from speechai.audio.io import AudioBuffer, generate_sine, pcm16_bytes
from speechai.audio.vad import EnergyVAD, StreamingVAD, build_vad


def _signal(silence_before: float, tone: float, silence_after: float, sr: int = 16000) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(int(silence_before * sr), np.float32),
            generate_sine(tone, sr, freq=300).samples,
            np.zeros(int(silence_after * sr), np.float32),
        ]
    )


def test_energy_vad_detects_tone() -> None:
    vad = EnergyVAD(threshold_db=-35.0)
    assert vad.is_speech(generate_sine(0.03, 16000).samples[:480]) is True
    assert vad.is_speech(np.zeros(480, np.float32)) is False


def test_build_vad_energy() -> None:
    vad = build_vad("energy")
    assert isinstance(vad, EnergyVAD)


def test_streaming_vad_isolates_utterance() -> None:
    vad = build_vad("energy")
    stream = StreamingVAD(vad, 16000, min_speech_ms=250, min_silence_ms=200)
    signal = _signal(0.5, 0.4, 0.5)
    completed: list = []
    for i in range(0, signal.size, 1600):
        completed.extend(stream.push(signal[i : i + 1600]))
    completed.extend(stream.flush())
    assert len(completed) == 1
    utterance = completed[0]
    assert utterance.start_time == pytest.approx(0.5, abs=0.05)
    assert utterance.duration == pytest.approx(0.4, abs=0.08)


def test_streaming_vad_gates_short_speech() -> None:
    vad = build_vad("energy")
    stream = StreamingVAD(vad, 16000, min_speech_ms=250, min_silence_ms=200)
    signal = _signal(0.1, 0.1, 0.1)  # 100 ms tone < min_speech_ms
    completed: list = []
    for i in range(0, signal.size, 1600):
        completed.extend(stream.push(signal[i : i + 1600]))
    completed.extend(stream.flush())
    assert len(completed) == 0


def test_streaming_vad_flush_closes_open_utterance() -> None:
    vad = build_vad("energy")
    stream = StreamingVAD(vad, 16000, min_speech_ms=250, min_silence_ms=200)
    signal = _signal(0.2, 0.5, 0.0)  # speech runs to the very end
    completed: list = []
    for i in range(0, signal.size, 1600):
        completed.extend(stream.push(signal[i : i + 1600]))
    assert len(completed) == 0
    completed.extend(stream.flush())
    assert len(completed) == 1


def test_pcm_feed_through_vad() -> None:
    vad = build_vad("energy")
    stream = StreamingVAD(vad, 16000, min_speech_ms=250, min_silence_ms=200)
    signal = _signal(0.4, 0.5, 0.4)
    pcm = pcm16_bytes(signal)
    finals = 0
    for i in range(0, len(pcm), 4800):
        audio = AudioBuffer.from_pcm16(pcm[i : i + 4800], 16000)
        finals += len(stream.push(audio.samples))
    finals += len(stream.flush())
    assert finals == 1
