"""Voice activity detection.

Two frame-level VADs are provided:

- :class:`WebRTCVAD` - production-grade WebRTC VAD (fast, tuned, no deps on
  heavy ML stacks; needs 16 kHz 10/20/30 ms frames).
- :class:`EnergyVAD` - dependency-free energy threshold fallback.

:class:`StreamingVAD` wraps either frame VAD and segments a continuous PCM
stream into utterances (speech regions) with hysteresis: speech must persist
for ``min_speech_ms`` to open an utterance and silence for ``min_silence_ms``
to close it. A hard ``max_utterance_ms`` cap protects against unbounded
utterances in call-center recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speechai.audio.io import pcm16_bytes
from speechai.core.errors import EngineUnavailableError


@dataclass
class Utterance:
    """A contiguous region of speech detected by the streaming VAD."""

    start_time: float
    end_time: float
    samples: np.ndarray  # float32 mono @ sample_rate
    is_partial: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class EnergyVAD:
    """Frame-energy based VAD. Robust, dependency-free."""

    def __init__(self, threshold_db: float = -35.0) -> None:
        self.threshold_db = threshold_db

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-12)
        db = 20.0 * np.log10(rms)
        return bool(db > self.threshold_db)


class WebRTCVAD:
    """WebRTC VAD wrapper (requires ``webrtcvad``; part of the engines extra)."""

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20 or 30 for WebRTC VAD")
        try:
            import webrtcvad
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EngineUnavailableError(
                "webrtcvad is not installed - set VAD backend to 'energy' or "
                "install the engines extra (pip install -e '.[engines]')"
            ) from exc
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_ms = frame_ms
        self._frame_bytes = int(16000 * frame_ms / 1000) * 2

    def is_speech(self, frame: np.ndarray) -> bool:
        data = pcm16_bytes(frame)
        if len(data) != self._frame_bytes:
            data = _pad_or_trim(data, self._frame_bytes)
        return bool(self._vad.is_speech(data, 16000))


def _pad_or_trim(data: bytes, length: int) -> bytes:
    if len(data) >= length:
        return data[:length]
    return data + b"\x00" * (length - len(data))


def build_vad(
    backend: str = "auto",
    *,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    threshold_db: float = -35.0,
) -> EnergyVAD | WebRTCVAD:
    """Factory for a frame-level VAD. ``auto`` prefers WebRTC, falls back to energy."""
    if backend == "webrtc":
        return WebRTCVAD(aggressiveness, frame_ms)
    if backend == "energy":
        return EnergyVAD(threshold_db)
    if backend == "auto":
        try:
            return WebRTCVAD(aggressiveness, frame_ms)
        except EngineUnavailableError:
            return EnergyVAD(threshold_db)
    raise ValueError(f"Unknown VAD backend: {backend}")


class StreamingVAD:
    """Segments a stream of PCM samples into utterances."""

    def __init__(
        self,
        vad: EnergyVAD | WebRTCVAD,
        sample_rate: int = 16000,
        *,
        frame_ms: int = 30,
        min_speech_ms: int = 250,
        min_silence_ms: int = 400,
        max_utterance_ms: int = 12000,
    ) -> None:
        self._vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_utterance_ms = max_utterance_ms
        self._frame_size = int(sample_rate * frame_ms / 1000)
        self._pending = np.empty(0, dtype=np.float32)
        self._in_utterance = False
        self._utterance = np.empty(0, dtype=np.float32)
        self._utterance_start_ms = 0
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._now_ms = 0

    # ------------------------------------------------------------------
    def push(self, chunk: np.ndarray) -> list[Utterance]:
        """Feed a chunk of mono float32 samples; returns completed utterances."""
        if chunk.size == 0:
            return []
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32)])
        completed: list[Utterance] = []
        while self._pending.size >= self._frame_size:
            frame = self._pending[: self._frame_size]
            self._pending = self._pending[self._frame_size :]
            self._now_ms += self.frame_ms
            utterance = self._process_frame(frame)
            if utterance is not None:
                completed.append(utterance)
        return completed

    def current_partial(self) -> Utterance | None:
        """The in-progress utterance buffer (for partial transcription)."""
        if not self._in_utterance or self._utterance.size == 0:
            return None
        return Utterance(
            start_time=self._utterance_start_ms / 1000.0,
            end_time=self._now_ms / 1000.0,
            samples=self._utterance,
            is_partial=True,
        )

    def flush(self) -> list[Utterance]:
        """Close any open utterance at end-of-stream."""
        if not self._in_utterance:
            return []
        end_time = self._now_ms / 1000.0
        utterance = Utterance(
            start_time=self._utterance_start_ms / 1000.0,
            end_time=end_time,
            samples=self._utterance,
        )
        self._in_utterance = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._utterance = np.empty(0, dtype=np.float32)
        return [utterance]

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> Utterance | None:
        speech = self._vad.is_speech(frame)
        if speech:
            self._silence_run_ms = 0
            self._speech_run_ms += self.frame_ms
            if not self._in_utterance and self._speech_run_ms >= self.min_speech_ms:
                self._in_utterance = True
                self._utterance_start_ms = self._now_ms - self._speech_run_ms
                self._utterance = np.empty(0, dtype=np.float32)
            if self._in_utterance:
                self._utterance = np.concatenate([self._utterance, frame])
                if (self._now_ms - self._utterance_start_ms) >= self.max_utterance_ms:
                    return self._close_utterance(trim_trailing_silence=False)
            return None
        # Silence
        self._speech_run_ms = 0
        if self._in_utterance:
            self._silence_run_ms += self.frame_ms
            self._utterance = np.concatenate([self._utterance, frame])
            if self._silence_run_ms >= self.min_silence_ms:
                return self._close_utterance(trim_trailing_silence=True)
        return None

    def _close_utterance(self, *, trim_trailing_silence: bool) -> Utterance:
        samples = self._utterance
        end_time = self._now_ms / 1000.0
        if trim_trailing_silence and self._silence_run_ms > 0:
            trim_samples = int(self._silence_run_ms / 1000.0 * self.sample_rate)
            if 0 < trim_samples < samples.size:
                samples = samples[:-trim_samples]
                end_time -= self._silence_run_ms / 1000.0
        utterance = Utterance(
            start_time=self._utterance_start_ms / 1000.0,
            end_time=max(end_time, self._utterance_start_ms / 1000.0),
            samples=samples,
        )
        self._in_utterance = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._utterance = np.empty(0, dtype=np.float32)
        return utterance
