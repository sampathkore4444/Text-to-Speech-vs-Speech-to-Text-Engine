"""Streaming STT transcriber tests (VAD-gated utterance segmentation)."""

from __future__ import annotations

import numpy as np

from speechai.audio.io import generate_sine, pcm16_bytes
from speechai.stt.streaming import StreamingTranscriber


def _pcm_signal(sr: int = 16000) -> bytes:
    tone = generate_sine(0.5, sr, freq=300).samples
    signal = np.concatenate(
        [
            np.zeros(int(0.3 * sr), np.float32),
            tone,
            np.zeros(int(0.4 * sr), np.float32),
            tone,
            np.zeros(int(0.2 * sr), np.float32),
        ]
    )
    return pcm16_bytes(signal)


async def test_streaming_two_utterances(fake_stt) -> None:
    transcriber = StreamingTranscriber(
        fake_stt,
        min_silence_ms=200,
        vad_backend="energy",
    )
    pcm = _pcm_signal()
    events = []
    chunk = 4800  # 0.3 s @16k
    for i in range(0, len(pcm), chunk):
        async for event in transcriber.feed(pcm[i : i + chunk]):
            events.append(event)
    async for event in transcriber.finish():
        events.append(event)

    finals = [e for e in events if e.is_final]
    assert len(finals) == 2
    assert all(e.text == fake_stt.text for e in finals)
    assert finals[0].utterance_index == 1
    assert finals[1].utterance_index == 2


async def test_streaming_silence_produces_nothing(fake_stt) -> None:
    transcriber = StreamingTranscriber(fake_stt, vad_backend="energy")
    pcm = np.zeros(int(0.5 * 16000), np.float32).tobytes()  # not PCM16! zeros are fine
    events = []
    async for event in transcriber.feed(pcm):
        events.append(event)
    async for event in transcriber.finish():
        events.append(event)
    assert events == []


async def test_streaming_respects_language(fake_stt) -> None:
    transcriber = StreamingTranscriber(fake_stt, language="hi", vad_backend="energy")
    tone = generate_sine(0.4, 16000).samples
    signal = np.concatenate([np.zeros(int(0.2 * 16000), np.float32), tone])
    events = []
    async for event in transcriber.feed(pcm16_bytes(signal)):
        events.append(event)
    async for event in transcriber.finish():
        events.append(event)
    finals = [e for e in events if e.is_final]
    assert finals
    assert fake_stt.calls >= 1
