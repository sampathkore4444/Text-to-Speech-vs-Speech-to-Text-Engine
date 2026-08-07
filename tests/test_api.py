"""API contract tests: REST endpoints, batch jobs flow, WebSockets, auth."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from speechai.api.app import create_app
from speechai.audio.io import generate_sine, pcm16_bytes
from speechai.core.config import Settings


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["queue"]["backend"] == "memory"


def test_metrics_exposed(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "stt_requests_total" in response.text
    assert "http_requests_total" in response.text


def test_request_id_header(client) -> None:
    response = client.get("/health")
    assert response.headers.get("x-request-id")


def test_demo_ui_served(client) -> None:
    """The browser demo console is served at the root."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Bank Speech AI" in response.text
    assert "Live microphone" in response.text
    assert "v1/ws/transcribe" in response.text


def test_openapi_includes_websocket_paths(client) -> None:
    """FastAPI omits WS routes from OpenAPI; the injected x-websocket paths must appear."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    body = response.json()
    paths = body["paths"]
    assert "/v1/ws/transcribe" in paths
    assert "/v1/ws/synthesize" in paths
    assert paths["/v1/ws/transcribe"]["get"]["x-websocket"] is True
    assert paths["/v1/ws/synthesize"]["get"]["x-websocket"] is True
    assert body["x-websocket-spec"] == "docs/ws-protocol.md"
    # The extension must not clobber the standard REST schema.
    assert "/v1/transcribe" in paths
    assert "post" in paths["/v1/transcribe"]


def test_metric_path_template_low_cardinality() -> None:
    from speechai.api.middleware import _template_path

    assert _template_path("/v1/jobs/abcd1234ef567890") == "/v1/jobs/{id}"
    assert _template_path("/v1/jobs/abcd1234ef567890/audio") == "/v1/jobs/{id}/audio"
    assert _template_path("/v1/transcribe") == "/v1/transcribe"
    assert _template_path("/v1/jobs/123456") == "/v1/jobs/{id}"


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
def test_transcribe_endpoint(client, sample_audio, fake_stt) -> None:
    with open(sample_audio, "rb") as fh:
        response = client.post(
            "/v1/transcribe", files={"file": ("sample.wav", fh, "audio/wav")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == fake_stt.text
    assert body["engine"] == "fake-stt"
    assert body["metrics"]["rtf"] >= 0


def test_transcribe_returns_sentence_segments(settings, fake_tts, sample_audio) -> None:
    """A whole-file single engine segment must come back as per-sentence rows."""
    from speechai.stt.base import Segment, TranscriptionResult

    class MultiSentenceSTT:
        name = "multi-stt"

        def load(self) -> None:
            pass

        def transcribe(self, audio, options=None):
            seg = Segment(text="First sentence. Second sentence!", start=0.0, end=6.0, confidence=0.9)
            return TranscriptionResult(
                text=seg.text,
                language="en",
                segments=[seg],
                duration_seconds=6.0,
                latency_seconds=0.1,
                rtf=0.02,
                engine=self.name,
                avg_confidence=0.9,
            )

        def close(self) -> None:
            pass

    app = create_app(settings, stt_engine=MultiSentenceSTT(), tts_engine=fake_tts)
    with TestClient(app) as client:
        with open(sample_audio, "rb") as fh:
            response = client.post(
                "/v1/transcribe", files={"file": ("sample.wav", fh, "audio/wav")}
            )
    assert response.status_code == 200
    body = response.json()
    assert [s["text"] for s in body["segments"]] == ["First sentence.", "Second sentence!"]
    assert body["segments"][0]["end"] <= body["segments"][1]["start"] + 1e-6
    assert body["segments"][1]["end"] == 6.0


def test_transcribe_vad_filter_form_param_reaches_engine(client, sample_audio, fake_stt) -> None:
    """The multipart vad_filter field must override the engine options."""
    with open(sample_audio, "rb") as fh:
        response = client.post(
            "/v1/transcribe",
            files={"file": ("sample.wav", fh, "audio/wav")},
            data={"vad_filter": "false"},
        )
    assert response.status_code == 200
    assert fake_stt.last_options is not None
    assert fake_stt.last_options.vad_filter is False


def test_transcribe_vad_filter_defaults_from_settings(client, sample_audio, fake_stt) -> None:
    """Omitted vad_filter falls back to settings.stt.vad_filter (default False)."""
    with open(sample_audio, "rb") as fh:
        response = client.post(
            "/v1/transcribe", files={"file": ("sample.wav", fh, "audio/wav")}
        )
    assert response.status_code == 200
    assert fake_stt.last_options is not None
    assert fake_stt.last_options.vad_filter is False


def test_transcribe_vad_filter_enable(client, sample_audio, fake_stt) -> None:
    """vad_filter=true must be forwarded as True."""
    with open(sample_audio, "rb") as fh:
        response = client.post(
            "/v1/transcribe",
            files={"file": ("sample.wav", fh, "audio/wav")},
            data={"vad_filter": "true"},
        )
    assert response.status_code == 200
    assert fake_stt.last_options is not None
    assert fake_stt.last_options.vad_filter is True


def test_transcribe_sync_vad_filter_wiring(pipeline, sample_audio, fake_stt) -> None:
    """Pipeline-level override: explicit value wins, omitted uses settings default."""
    pipeline.transcribe_sync(sample_audio, vad_filter=True)
    assert fake_stt.last_options is not None and fake_stt.last_options.vad_filter is True
    pipeline.transcribe_sync(sample_audio, vad_filter=False)
    assert fake_stt.last_options.vad_filter is False
    pipeline.transcribe_sync(sample_audio)
    assert fake_stt.last_options.vad_filter is False  # settings.stt.vad_filter default


def test_transcribe_redacts_pii(client, sample_audio, fake_stt) -> None:
    fake_stt.text = "My card is 4242 4242 4242 4242"
    with open(sample_audio, "rb") as fh:
        response = client.post(
            "/v1/transcribe", files={"file": ("sample.wav", fh, "audio/wav")}
        )
    body = response.json()
    assert body["redacted"] is True
    assert "4242 4242 4242 4242" not in body["text"]
    assert body["redactions"][0]["type"] == "card"


def test_transcribe_empty_file_rejected(client) -> None:
    response = client.post(
        "/v1/transcribe", files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_transcribe_invalid_audio(client) -> None:
    response = client.post(
        "/v1/transcribe", files={"file": ("bad.wav", b"not audio", "audio/wav")}
    )
    assert response.status_code in (400, 500)
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def test_synthesize_returns_wav(client, fake_tts) -> None:
    response = client.post("/v1/synthesize", json={"text": "Hello bank customer"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert fake_tts.calls == 1


def test_synthesize_validation(client) -> None:
    response = client.post("/v1/synthesize", json={"text": ""})
    assert response.status_code == 422


def test_synthesize_streaming(client, fake_tts) -> None:
    with client.stream("POST", "/v1/synthesize?stream=true", json={"text": "One. Two."}) as response:
        chunks = [c for c in response.iter_bytes()]
    assert b"".join(chunks)[:4] == b"RIFF"
    assert fake_tts.calls == 2


# ---------------------------------------------------------------------------
# Batch jobs flow (API + in-process worker)
# ---------------------------------------------------------------------------
def test_jobs_flow_with_worker(client, sample_audio, fake_stt) -> None:
    stop = threading.Event()
    errors: list[str] = []

    def worker_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pipeline = client.app.state.pipeline
        queue = client.app.state.queue

        async def consume() -> None:
            while not stop.is_set():
                job = await queue.dequeue(timeout=0.05)
                if job is not None:
                    await pipeline.run_job(job)

        loop.run_until_complete(consume())

    thread = threading.Thread(target=worker_loop, daemon=True)
    thread.start()
    try:
        with open(sample_audio, "rb") as fh:
            submitted = client.post(
                "/v1/jobs/transcribe", files={"file": ("sample.wav", fh, "audio/wav")}
            )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]
        assert submitted.json()["url"] == f"/v1/jobs/{job_id}"

        status = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = client.get(f"/v1/jobs/{job_id}").json()["status"]
            if status in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert status == "succeeded", f"job did not succeed (errors={errors})"
        body = client.get(f"/v1/jobs/{job_id}").json()
        assert body["result"]["text"] == fake_stt.text

        deleted = client.delete(f"/v1/jobs/{job_id}")
        assert deleted.status_code == 204
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404
    finally:
        stop.set()
        thread.join(timeout=5)


def test_job_not_found(client) -> None:
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------
def test_ws_transcribe_stream(client, fake_stt) -> None:
    sr = 16000
    tone = generate_sine(0.5, sr, freq=300).samples
    signal = np.concatenate(
        [np.zeros(int(0.3 * sr), np.float32), tone, np.zeros(int(0.3 * sr), np.float32)]
    )
    pcm = pcm16_bytes(signal)

    with client.websocket_connect("/v1/ws/transcribe") as ws:
        ws.send_json({"sample_rate": sr})
        for i in range(0, len(pcm), 4800):
            ws.send_bytes(pcm[i : i + 4800])
        ws.send_json({"action": "stop"})
        events = []
        while True:
            try:
                events.append(ws.receive_json())
            except Exception:
                break

    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"] == fake_stt.text


def test_ws_synthesize_stream(client, fake_tts) -> None:
    with client.websocket_connect("/v1/ws/synthesize") as ws:
        ws.send_json({"text": "Hello world. Goodbye world."})
        chunks: list[bytes] = []
        done = False
        while not done:
            message = ws.receive()
            # Different starlette versions use 'websocket.receive' or 'websocket.send'.
            if message["type"] in ("websocket.receive", "websocket.send"):
                if message.get("bytes"):
                    chunks.append(message["bytes"])
                elif message.get("text"):
                    done = json.loads(message["text"]).get("type") == "done"
            elif message["type"] == "websocket.disconnect":
                break
    assert len(chunks) == 2
    assert chunks[0][:4] == b"RIFF"


def test_ws_transcribe_fails_fast_on_missing_model(settings, fake_tts) -> None:
    """A missing STT model must surface as an error event, not a silent hang."""
    from starlette.websockets import WebSocketDisconnect

    from speechai.core.errors import ModelNotFoundError

    class BrokenSTT:
        name = "broken-stt"

        def load(self) -> None:
            raise ModelNotFoundError("Whisper model not found in test env")

        def transcribe(self, audio, options=None):  # pragma: no cover - never reached
            raise AssertionError("transcribe must not run when load fails")

        def close(self) -> None:
            pass

    app = create_app(settings, stt_engine=BrokenSTT(), tts_engine=fake_tts)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/transcribe") as ws:
            ws.send_json({"sample_rate": 16000})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "model_not_found"
            with pytest.raises(WebSocketDisconnect) as exc_info:
                while True:
                    ws.receive_json()
            assert exc_info.value.code == 1011


def test_ws_synthesize_fails_fast_on_missing_model(settings, fake_stt) -> None:
    """A missing TTS voice must surface as an error event, not a silent hang."""
    from starlette.websockets import WebSocketDisconnect

    from speechai.core.errors import ModelNotFoundError

    class BrokenTTS:
        name = "broken-tts"

        def load(self) -> None:
            raise ModelNotFoundError("Piper voice not found in test env")

        def synthesize(self, text, options=None):  # pragma: no cover - never reached
            raise AssertionError("synthesize must not run when load fails")

        def close(self) -> None:
            pass

    app = create_app(settings, stt_engine=fake_stt, tts_engine=BrokenTTS())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/synthesize") as ws:
            ws.send_json({"text": "Hello"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "model_not_found"
            with pytest.raises(WebSocketDisconnect) as exc_info:
                while True:
                    ws.receive_json()
            assert exc_info.value.code == 1011


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_api_key_enforced(tmp_path, fake_stt, fake_tts) -> None:
    settings = Settings(
        service={"name": "test", "environment": "development", "log_level": "WARNING", "log_format": "text"},
        storage={"data_dir": str(tmp_path / "data")},
        api={"api_key": "s3cret"},
        queue={"backend": "memory"},
        vad={"backend": "energy"},
    )
    app = create_app(settings, stt_engine=fake_stt, tts_engine=fake_tts)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200  # whitelisted
        assert client.get("/metrics").status_code == 200  # whitelisted
        assert client.get("/").status_code == 200  # demo UI whitelisted
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
