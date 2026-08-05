"""Streaming TTS synthesizer tests."""

from __future__ import annotations

from speechai.tts.streaming import StreamingSynthesizer


async def test_chunks_by_sentence(fake_tts) -> None:
    synthesizer = StreamingSynthesizer(fake_tts)
    chunks = [c async for c in synthesizer.synthesize("Hello world. My name is Ada.")]
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk[:4] == b"RIFF"
    assert fake_tts.calls == 2


async def test_single_sentence(fake_tts) -> None:
    synthesizer = StreamingSynthesizer(fake_tts)
    chunks = [c async for c in synthesizer.synthesize("One sentence only")]
    assert len(chunks) == 1


async def test_synthesize_full(fake_tts) -> None:
    synthesizer = StreamingSynthesizer(fake_tts)
    full = await synthesizer.synthesize_full("First sentence. Second sentence.")
    assert full[:4] == b"RIFF"


async def test_empty_text_synthesizes_one_chunk(fake_tts) -> None:
    synthesizer = StreamingSynthesizer(fake_tts)
    chunks = [c async for c in synthesizer.synthesize("")]
    assert len(chunks) == 1
