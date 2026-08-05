"""Segment-based streaming transcription.

Real-time ASR with a local Whisper model is delivered utterance-by-utterance:
the incoming 16 kHz PCM stream is split into utterances by the streaming VAD;
each completed utterance is transcribed in a worker thread and emitted as a
*final* event. While an utterance is in progress, *partial* events are emitted
every ``partial_interval_ms`` so clients can show live hypotheses.

This pattern keeps latency low and quality high (no context bleed between
utterances) and is a well-known production approach for local Whisper
deployments.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from speechai.audio.io import ASR_SAMPLE_RATE, AudioBuffer
from speechai.audio.vad import StreamingVAD, Utterance, build_vad
from speechai.stt.base import Segment, STTEngine, STTOptions
from speechai.stt.postprocess import TextPostProcessor


@dataclass
class StreamEvent:
    """One streaming transcription result (partial or final)."""

    is_final: bool
    text: str
    start: float
    end: float
    utterance_index: int
    segments: list[Segment] | None = None
    confidence: float | None = None

    def to_dict(self) -> dict:
        data = {
            "type": "final" if self.is_final else "partial",
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "utterance_index": self.utterance_index,
            "confidence": self.confidence,
        }
        if self.segments is not None:
            data["segments"] = [
                {"text": s.text, "start": s.start, "end": s.end, "confidence": s.confidence}
                for s in self.segments
            ]
        return data


class StreamingTranscriber:
    """VAD-gated streaming transcriber. Not thread-safe; one instance per connection."""

    def __init__(
        self,
        engine: STTEngine,
        *,
        language: str | None = None,
        input_rate: int = ASR_SAMPLE_RATE,
        partial_interval_ms: int = 2500,
        max_utterance_ms: int = 12000,
        min_silence_ms: int = 500,
        vad_backend: str = "auto",
        postprocessor: TextPostProcessor | None = None,
    ) -> None:
        self.engine = engine
        self.language = language
        self.input_rate = input_rate
        self.partial_interval_ms = partial_interval_ms
        self.postprocessor = postprocessor or TextPostProcessor()
        frame_vad = build_vad(vad_backend)
        self._vad = StreamingVAD(
            frame_vad,
            ASR_SAMPLE_RATE,
            min_silence_ms=min_silence_ms,
            max_utterance_ms=max_utterance_ms,
        )
        self._utterance_index = 0
        self._last_partial_at = 0.0
        self._stream_started = time.monotonic()

    # ------------------------------------------------------------------
    async def feed(self, pcm16_chunk: bytes) -> AsyncIterator[StreamEvent]:
        """Feed raw PCM16 audio; yields partial/final events as speech completes."""
        audio = AudioBuffer.from_pcm16(pcm16_chunk, self.input_rate)
        if audio.sample_rate != ASR_SAMPLE_RATE:
            audio = audio.resample(ASR_SAMPLE_RATE)
        for utterance in self._vad.push(audio.samples):
            yield await self._transcribe_utterance(utterance, final=True)
        partial = self._vad.current_partial()
        if partial is not None:
            elapsed = time.monotonic() - self._stream_started
            if elapsed - self._last_partial_at >= self.partial_interval_ms / 1000.0:
                self._last_partial_at = elapsed
                yield await self._transcribe_utterance(partial, final=False)

    async def finish(self) -> AsyncIterator[StreamEvent]:
        """Close the stream and emit any remaining final utterance."""
        for utterance in self._vad.flush():
            yield await self._transcribe_utterance(utterance, final=True)

    # ------------------------------------------------------------------
    async def _transcribe_utterance(self, utterance: Utterance, *, final: bool) -> StreamEvent:
        audio = AudioBuffer(utterance.samples, ASR_SAMPLE_RATE)
        options = STTOptions(language=self.language)
        result = await asyncio.to_thread(self.engine.transcribe, audio, options)
        if final:
            processed = self.postprocessor.process(result.text)
            text, segments = processed.text, result.segments
        else:
            # Partials: clean but do not redact yet (avoids flickering masks).
            text = result.text.strip()
            segments = None
        if final:
            self._utterance_index += 1
        return StreamEvent(
            is_final=final,
            text=text,
            start=utterance.start_time,
            end=utterance.end_time,
            utterance_index=self._utterance_index,
            segments=segments,
            confidence=result.avg_confidence,
        )
