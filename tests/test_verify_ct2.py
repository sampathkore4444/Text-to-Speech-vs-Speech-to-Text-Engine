"""Tests for ``scripts/verify_ct2_model.py`` (int8 CT2 vs fp32 baseline gate).

The script is loaded via importlib (``scripts/`` is not a package); its pure
functions need no model files, and ``main()`` is tested by stubbing
``run_from_manifest`` so no faster-whisper model is loaded.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from speechai.audio.io import generate_sine, write_wav
from speechai.core.config import Settings
from speechai.eval.metrics import EvaluationReport, UtteranceResult

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_ct2_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_ct2_model", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load_script()


def _fp32_payload(wer_mean: float = 0.04, *, utterances: list[dict] | None = None) -> dict:
    return {
        "aggregates": {
            "wer": {"mean": wer_mean, "median": wer_mean, "p90": wer_mean},
            "cer": {"mean": 0.02, "median": 0.02, "p90": 0.02},
            "rtf": {"mean": 0.20, "median": 0.20, "p90": 0.20},
            "latency_seconds": {"mean": 1.0, "median": 1.0, "p90": 1.0},
        },
        "utterances": utterances or [],
    }


def _make_report(*, wer: float, rtf: float = 0.2) -> EvaluationReport:
    return EvaluationReport(
        engine="faster-whisper",
        dataset="eval-set",
        results=[
            UtteranceResult(
                audio="data/samples/sample_01.wav",
                reference="the reference",
                hypothesis="the hyp",
                wer=wer,
                cer=0.05,
                audio_duration=3.0,
                latency_seconds=0.6,
                rtf=rtf,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Pure gate logic
# ---------------------------------------------------------------------------
def test_compute_wer_gap():
    assert verify.compute_wer_gap(0.10, 0.04) == pytest.approx(0.06)
    assert verify.compute_wer_gap(0.03, 0.04) == pytest.approx(-0.01)


def test_gather_problems_no_problems():
    problems = verify.gather_problems(
        0.05, 0.30, 0.04, max_wer_gap=0.05, max_wer_abs=0.10, max_rtf=0.50
    )
    assert problems == []


def test_gather_problems_gap_breach():
    problems = verify.gather_problems(
        0.12, 0.30, 0.04, max_wer_gap=0.05, max_wer_abs=0.10, max_rtf=0.50
    )
    assert any("quantization gap" in p for p in problems)


def test_gather_problems_abs_breach():
    problems = verify.gather_problems(
        0.20, 0.30, 0.04, max_wer_gap=0.05, max_wer_abs=0.10, max_rtf=0.50
    )
    assert any("absolute bar" in p for p in problems)


def test_gather_problems_rtf_breach():
    problems = verify.gather_problems(
        0.05, 0.80, 0.04, max_wer_gap=0.05, max_wer_abs=0.10, max_rtf=0.50
    )
    assert any("RTF" in p and "slow" in p for p in problems)


def test_top_regressions_only_worse_utterances():
    # Keys are resolved absolute paths (mirrors load_fp32_report).
    fp32_map = {
        str(Path("data/samples/sample_01.wav").resolve()): {"wer": 0.0, "hypothesis": "perfect"},
    }
    report = _make_report(wer=0.33)
    rows = verify.top_regressions(report, fp32_map)
    assert len(rows) == 1
    assert rows[0]["wer_delta"] == pytest.approx(0.33)

    report_flat = _make_report(wer=0.0)
    assert verify.top_regressions(report_flat, fp32_map) == []


# ---------------------------------------------------------------------------
# fp32 report loading
# ---------------------------------------------------------------------------
def test_load_fp32_report(tmp_path: Path):
    report_file = tmp_path / "report_finetuned.json"
    report_file.write_text(json.dumps(_fp32_payload(0.04, utterances=[{"audio": "x.wav", "wer": 0.1}])), encoding="utf-8")
    loaded = verify.load_fp32_report(report_file)
    assert loaded["aggregates"]["wer"]["mean"] == pytest.approx(0.04)
    assert str(Path("x.wav").resolve()) in loaded["utterances"]


def test_load_fp32_report_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        verify.load_fp32_report(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# main() integration (engine stubbed - no faster-whisper required)
# ---------------------------------------------------------------------------
def _write_fixtures(tmp_path: Path, *, fp32_wer: float = 0.04) -> tuple[str, str, str]:
    ct2_dir = tmp_path / "ct2"
    ct2_dir.mkdir()
    (ct2_dir / "model.bin").write_bytes(b"fake")
    manifest = tmp_path / "eval_manifest.jsonl"
    manifest.write_text(json.dumps({"audio": "data/samples/sample_01.wav", "reference": "the reference"}) + "\n", encoding="utf-8")
    fp32_report = tmp_path / "report_finetuned.json"
    fp32_report.write_text(json.dumps(_fp32_payload(fp32_wer)), encoding="utf-8")
    return str(ct2_dir), str(manifest), str(fp32_report)


def test_main_pass(tmp_path: Path, monkeypatch):
    ct2_dir, manifest, fp32_report = _write_fixtures(tmp_path)
    monkeypatch.setattr(verify, "run_from_manifest", lambda engine, m, language=None: _make_report(wer=0.05))
    report_out = tmp_path / "verify.json"
    exit_code = verify.main(
        [
            "--ct2-dir", ct2_dir, "--manifest", manifest, "--fp32-report", fp32_report,
            "--report", str(report_out),
            "--no-batch-check", "--no-ws-check",
        ]
    )
    assert exit_code == 0
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["problems"] == []
    assert payload["wer_gap"] == pytest.approx(0.01)
    assert payload["spot_checks"] == {}


def test_main_fails_when_gap_breached(tmp_path: Path, monkeypatch):
    ct2_dir, manifest, fp32_report = _write_fixtures(tmp_path, fp32_wer=0.04)
    monkeypatch.setattr(verify, "run_from_manifest", lambda engine, m, language=None: _make_report(wer=0.20))
    report_out = tmp_path / "verify.json"
    exit_code = verify.main(
        [
            "--ct2-dir", ct2_dir, "--manifest", manifest, "--fp32-report", fp32_report,
            "--report", str(report_out),
            "--no-batch-check", "--no-ws-check",
        ]
    )
    assert exit_code == 1
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert any("quantization gap" in p for p in payload["problems"])


def test_main_missing_ct2_dir(tmp_path: Path):
    ct2_dir, manifest, fp32_report = _write_fixtures(tmp_path)
    exit_code = verify.main(
        [
            "--ct2-dir", str(tmp_path / "missing"), "--manifest", manifest,
            "--fp32-report", fp32_report,
        ]
    )
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Serving-path spot checks (batch job + WebSocket), tested with a fake engine
# against the real FastAPI app - no faster-whisper needed.
# ---------------------------------------------------------------------------
class _FakeSpotSTT:
    name = "fake-spot-stt"

    def load(self) -> None:
        pass

    def transcribe(self, audio, options=None):
        from speechai.stt.base import Segment, TranscriptionResult

        segment = Segment(text="hello bank customer", start=0.0, end=1.0, confidence=0.9)
        return TranscriptionResult(
            text=segment.text,
            language="en",
            segments=[segment],
            duration_seconds=1.0,
            latency_seconds=0.01,
            rtf=0.01,
            engine=self.name,
            avg_confidence=0.9,
        )

    def close(self) -> None:
        pass


def _spot_settings(tmp_path: Path) -> Settings:
    return Settings(
        service={"name": "verify-test", "environment": "development", "log_level": "WARNING", "log_format": "text"},
        storage={"data_dir": str(tmp_path / "data")},
        queue={"backend": "memory"},
        vad={"backend": "energy"},
    )


def test_check_batch_path_ok(tmp_path: Path):
    """A transcription job runs end to end through the real REST batch path."""
    settings = _spot_settings(tmp_path)
    audio = tmp_path / "tone.wav"
    write_wav(audio, generate_sine(1.0, 16000))
    result = verify._check_batch_path(settings, audio, stt_engine=_FakeSpotSTT())
    assert result["ok"] is True
    assert result["status"] == "succeeded"
    assert result["text_length"] == len("hello bank customer")


def test_check_ws_path_ok(tmp_path: Path):
    """Streaming PCM through /v1/ws/transcribe yields a final event."""
    settings = _spot_settings(tmp_path)
    audio = tmp_path / "tone.wav"
    write_wav(audio, generate_sine(0.5, 16000, freq=300))
    result = verify._check_ws_path(settings, audio, stt_engine=_FakeSpotSTT())
    assert result["ok"] is True
    assert result["finals"] >= 1


def test_main_records_spot_checks(tmp_path: Path, monkeypatch):
    """Passing spot checks land in the report and keep the run green."""
    ct2_dir, manifest, fp32_report = _write_fixtures(tmp_path)
    monkeypatch.setattr(verify, "run_from_manifest", lambda engine, m, language=None: _make_report(wer=0.05))
    monkeypatch.setattr(
        verify, "_check_batch_path",
        lambda *a, **k: {"ok": True, "status": "succeeded", "text_length": 7},
    )
    monkeypatch.setattr(
        verify, "_check_ws_path",
        lambda *a, **k: {"ok": True, "events": 2, "finals": 1},
    )
    report_out = tmp_path / "verify.json"
    exit_code = verify.main(
        [
            "--ct2-dir", ct2_dir, "--manifest", manifest, "--fp32-report", fp32_report,
            "--report", str(report_out),
        ]
    )
    assert exit_code == 0
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["spot_checks"]["batch"]["ok"] is True
    assert payload["spot_checks"]["ws"]["ok"] is True


def test_main_spot_check_failure_gates_run(tmp_path: Path, monkeypatch):
    """A failed serving-path spot check must block the swap (exit 1)."""
    ct2_dir, manifest, fp32_report = _write_fixtures(tmp_path)
    monkeypatch.setattr(verify, "run_from_manifest", lambda engine, m, language=None: _make_report(wer=0.05))
    monkeypatch.setattr(
        verify, "_check_batch_path",
        lambda *a, **k: {"ok": False, "error": "submit returned HTTP 500"},
    )
    monkeypatch.setattr(
        verify, "_check_ws_path",
        lambda *a, **k: {"ok": True, "events": 1, "finals": 1},
    )
    report_out = tmp_path / "verify.json"
    exit_code = verify.main(
        [
            "--ct2-dir", ct2_dir, "--manifest", manifest, "--fp32-report", fp32_report,
            "--report", str(report_out),
        ]
    )
    assert exit_code == 1
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert any("batch serving path" in p for p in payload["problems"])
    assert payload["spot_checks"]["batch"]["ok"] is False
