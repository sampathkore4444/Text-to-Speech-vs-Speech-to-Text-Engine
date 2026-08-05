"""Shared test fixtures: settings, stub engines, sample audio, API client."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from speechai.api.app import create_app
from speechai.audio.io import generate_sine, write_wav
from speechai.core.config import Settings
from speechai.stt.base import Segment, STTOptions, TranscriptionResult
from speechai.tts.base import SynthesisResult, TTSOptions


class FakeSTTEngine:
    """Deterministic STT engine for tests."""

    name = "fake-stt"

    def __init__(self, text: str = "The quick brown fox jumps over the lazy dog.") -> None:
        self.text = text
        self.latency = 0.005
        self.calls = 0

    def load(self) -> None:
        pass

    def transcribe(self, audio, options: STTOptions | None = None) -> TranscriptionResult:
        self.calls += 1
        time.sleep(self.latency)
        segment = Segment(text=self.text, start=0.0, end=audio.duration_seconds, confidence=0.95)
        language = (options.language if options else None) or "en"
        return TranscriptionResult(
            text=self.text,
            language=language,
            segments=[segment],
            duration_seconds=audio.duration_seconds,
            latency_seconds=self.latency,
            rtf=self.latency / max(audio.duration_seconds, 1e-6),
            engine=self.name,
            avg_confidence=0.95,
        )

    def close(self) -> None:
        pass


class FakeTTSEngine:
    """Deterministic TTS engine that returns a short sine tone."""

    name = "fake-tts"

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None:
        pass

    def synthesize(self, text: str, options: TTSOptions | None = None) -> SynthesisResult:
        self.calls += 1
        audio = generate_sine(0.2, 22050)
        return SynthesisResult(
            audio=audio,
            latency_seconds=0.01,
            rtf=0.05,
            chars=len(text),
            engine=self.name,
        )

    def close(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        service={"name": "test", "environment": "development", "log_level": "WARNING", "log_format": "text"},
        storage={"data_dir": str(tmp_path / "data"), "result_ttl_seconds": 3600, "max_results": 100},
        queue={"backend": "memory", "poll_interval_seconds": 0.01},
        vad={"backend": "energy"},
        stt={"model_size": "tiny", "device": "cpu", "compute_type": "int8"},
        tts={"voice": "en_US-lessac-medium", "model_path": str(tmp_path / "fake.onnx")},
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def fake_stt() -> FakeSTTEngine:
    return FakeSTTEngine()


@pytest.fixture
def fake_tts() -> FakeTTSEngine:
    return FakeTTSEngine()


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """A short synthetic audio file with silence around a tone."""
    import numpy as np

    sr = 16000
    tone = generate_sine(0.4, sr, freq=300).samples
    signal = np.concatenate([np.zeros(int(0.2 * sr), np.float32), tone])
    path = tmp_path / "sample.wav"
    write_wav(path, generate_sine(0.5, 16000))
    _ = signal
    return path


@pytest.fixture
def client(settings: Settings, fake_stt: FakeSTTEngine, fake_tts: FakeTTSEngine):
    app = create_app(settings, stt_engine=fake_stt, tts_engine=fake_tts)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def pipeline(settings: Settings, fake_stt: FakeSTTEngine, fake_tts: FakeTTSEngine):
    from speechai.pipeline.batch import BatchPipeline
    from speechai.pipeline.queue import build_queue

    queue = build_queue(settings)
    pipe = BatchPipeline(settings, queue, stt_engine=fake_stt, tts_engine=fake_tts)
    yield pipe
    await queue.close()
