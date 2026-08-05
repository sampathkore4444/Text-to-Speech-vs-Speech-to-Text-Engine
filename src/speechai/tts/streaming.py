"""Chunked (sentence-by-sentence) synthesis for low-latency TTS streaming.

Synthesizing one sentence at a time lets clients start playback while the
rest of the message is still being generated - a key pattern for interactive
voice assistants (IVR) in banking.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from speechai.tts.base import TTSEngine, TTSOptions
from speechai.tts.textnorm import split_sentences


class StreamingSynthesizer:
    """Yields one WAV blob per sentence."""

    def __init__(self, engine: TTSEngine, *, speed: float = 1.0) -> None:
        self.engine = engine
        self.speed = speed

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        sentences = split_sentences(text) or [text]
        for sentence in sentences:
            result = await asyncio.to_thread(
                self.engine.synthesize, sentence, TTSOptions(speed=self.speed)
            )
            yield result.audio.to_wav_bytes()

    async def synthesize_full(self, text: str) -> bytes:
        """Convenience: all sentences stitched into a single WAV payload."""
        parts = [chunk async for chunk in self.synthesize(text)]
        return b"".join(parts)
