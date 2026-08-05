"""Timing helpers used by the inference pipelines and the evaluation harness."""

from __future__ import annotations

import time


class Stopwatch:
    """Monotonic wall-clock stopwatch."""

    __slots__ = ("start",)

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def reset(self) -> None:
        self.start = time.perf_counter()


def compute_rtf(processing_seconds: float, audio_seconds: float) -> float:
    """Real-time factor: how many seconds of compute per second of audio.

    RTF < 1.0 means faster than real-time. Returns 0.0 when audio duration
    is unknown or zero to avoid division errors.
    """
    if audio_seconds <= 0:
        return 0.0
    return processing_seconds / audio_seconds
