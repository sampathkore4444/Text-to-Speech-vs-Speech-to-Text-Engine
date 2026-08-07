"""Experiment tracking tests: no-op fallbacks + fake-MLflow stub paths.

These run without MLflow installed - the enabled paths use an in-process stub
injected via ``sys.modules``, and the disabled paths must no-op regardless.
"""

from __future__ import annotations

import sys
import types

import pytest

from speechai.core.config import Settings
from speechai.core.tracking import ExperimentTracker


class FakeMlflow:
    """Records every call so tests can assert what the tracker sent."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.artifacts: list[tuple[str, str]] = []
        self.finished: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name=None) -> None:
        self.calls.append(("start_run", run_name))

    def set_tags(self, tags: dict) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step=None) -> None:
        for key, value in metrics.items():
            self.metrics[key] = (value, step)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(("file", path))

    def log_artifacts(self, path: str) -> None:
        self.artifacts.append(("dir", path))

    def end_run(self, status=None) -> None:
        self.finished = status


@pytest.fixture
def fake_mlflow(monkeypatch) -> FakeMlflow:
    stub = FakeMlflow()
    module = types.ModuleType("mlflow")
    for name in (
        "set_tracking_uri", "set_experiment", "start_run", "set_tags",
        "log_params", "log_metrics", "log_artifact", "log_artifacts", "end_run",
    ):
        setattr(module, name, getattr(stub, name))
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return stub


# ---------------------------------------------------------------------------
# Disabled / unavailable paths - must silently no-op
# ---------------------------------------------------------------------------
def test_tracker_noop_without_uri() -> None:
    tracker = ExperimentTracker(enabled=True, tracking_uri=None, experiment_name="t")
    assert tracker.enabled is False
    tracker.start(tags={"a": "b"})
    tracker.log_params({"x": 1})
    tracker.log_metrics({"y": 0.5})
    tracker.log_artifact("nope.json")
    tracker.end()


def test_tracker_noop_when_mlflow_missing(monkeypatch) -> None:
    # Setting the sys.modules entry to None makes ``import mlflow`` raise
    # ImportError, deterministically simulating "not installed".
    monkeypatch.setitem(sys.modules, "mlflow", None)
    tracker = ExperimentTracker(enabled=True, tracking_uri="http://mlflow.test", experiment_name="t")
    assert tracker.enabled is False  # no crash, just disabled
    tracker.end()


def test_tracker_noop_when_explicitly_disabled(fake_mlflow) -> None:
    tracker = ExperimentTracker(enabled=False, tracking_uri="http://mlflow.test", experiment_name="t")
    assert tracker.enabled is False
    tracker.log_params({"x": 1})
    assert fake_mlflow.params == {}


# ---------------------------------------------------------------------------
# Enabled path through the stub
# ---------------------------------------------------------------------------
def test_tracker_logs_params_metrics_and_artifacts(fake_mlflow, tmp_path) -> None:
    tracker = ExperimentTracker(
        enabled=True, tracking_uri="http://mlflow.test", experiment_name="exp",
        run_name="my-run",
    )
    assert tracker.enabled is True
    tracker.start(tags={"task": "finetune"})
    # None must be dropped, tuples coerced to str.
    tracker.log_params({"epochs": 3, "base_model": "openai/whisper-tiny", "none_val": None, "pair": (1, 2)})
    tracker.log_metrics({"wer": 0.31, "none_metric": None}, step=7)

    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "a.txt").write_text("a", encoding="utf-8")
    tracker.log_artifact(report)
    tracker.log_artifact(sub)
    tracker.end()

    assert ("set_tracking_uri", "http://mlflow.test") in fake_mlflow.calls
    assert ("set_experiment", "exp") in fake_mlflow.calls
    assert ("start_run", "my-run") in fake_mlflow.calls
    assert fake_mlflow.tags == {"task": "finetune"}
    assert fake_mlflow.params == {"epochs": 3, "base_model": "openai/whisper-tiny", "pair": "(1, 2)"}
    assert fake_mlflow.metrics == {"wer": (0.31, 7)}
    assert ("file", str(report)) in fake_mlflow.artifacts
    assert ("dir", str(sub)) in fake_mlflow.artifacts
    assert fake_mlflow.finished == "FINISHED"


def test_tracker_uses_env_uri_fallback(monkeypatch, fake_mlflow) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow.test")
    tracker = ExperimentTracker(enabled=True, tracking_uri=None, experiment_name="exp")
    assert tracker.enabled is True
    tracker.end()


def test_tracker_missing_artifact_is_safe(fake_mlflow) -> None:
    tracker = ExperimentTracker(enabled=True, tracking_uri="http://mlflow.test", experiment_name="exp")
    tracker.start()
    tracker.log_artifact("does/not/exist.json")  # warning, not an exception
    assert fake_mlflow.artifacts == []
    tracker.end()


# ---------------------------------------------------------------------------
# speechai evaluate integration
# ---------------------------------------------------------------------------
def test_evaluate_tracks_through_stub(fake_mlflow, settings, tmp_path) -> None:
    from speechai.cli.main import _track_evaluation
    from speechai.eval.metrics import EvaluationReport, UtteranceResult

    settings.tracking.enabled = True
    settings.tracking.tracking_uri = "http://mlflow.test"
    settings.tracking.experiment_name = "speechai-eval"

    report = EvaluationReport(
        engine="fake-stt",
        dataset="calls",
        results=[
            UtteranceResult(
                audio="a.wav", reference="r", hypothesis="h",
                wer=0.1, cer=0.05, audio_duration=2.0, latency_seconds=0.5, rtf=0.25,
            )
        ],
    )
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    tracker = _track_evaluation(settings, report, report_path, enabled=True)
    assert tracker is not None
    tracker.end()

    assert fake_mlflow.params["dataset"] == "calls"
    assert fake_mlflow.params["engine"] == "fake-stt"
    assert fake_mlflow.params["model"] == settings.stt.model_size
    assert fake_mlflow.metrics["wer_mean"][0] == 0.1
    assert fake_mlflow.metrics["rtf_p90"][0] == 0.25
    assert ("file", str(report_path)) in fake_mlflow.artifacts
    assert fake_mlflow.finished == "FINISHED"


def test_cli_evaluate_wires_tracking_end_to_end(monkeypatch, fake_mlflow, tmp_path) -> None:
    """`speechai evaluate` records a run through the real CLI dispatch path."""
    from speechai.cli import main as cli_main
    from speechai.core.config import Settings
    from speechai.eval.metrics import EvaluationReport, UtteranceResult

    report = EvaluationReport(
        engine="fake-stt",
        dataset="calls",
        results=[
            UtteranceResult(
                audio="a.wav", reference="r", hypothesis="h",
                wer=0.2, cer=0.1, audio_duration=1.0, latency_seconds=0.4, rtf=0.4,
            )
        ],
    )

    def fake_run(engine, manifest, language=None):
        assert manifest == "data/manifest.jsonl"
        return report

    monkeypatch.setattr(cli_main, "run_from_manifest", fake_run)
    monkeypatch.setattr(cli_main, "build_stt_engine", lambda settings: object())
    monkeypatch.setattr(
        Settings,
        "load",
        staticmethod(
            lambda cls=None: Settings(
                tracking={"enabled": True, "tracking_uri": "http://mlflow.test"},
                eval={"report_dir": str(tmp_path)},
                stt={"model_size": "tiny"},
            )
        ),
    )

    exit_code = cli_main.main(["evaluate", "data/manifest.jsonl", "--report", str(tmp_path / "r.json")])
    assert exit_code == 0
    assert fake_mlflow.params["dataset"] == "calls"
    assert fake_mlflow.metrics["wer_mean"][0] == 0.2
    assert fake_mlflow.finished == "FINISHED"


def test_cli_evaluate_marks_run_failed_on_gate_breach(monkeypatch, fake_mlflow, tmp_path) -> None:
    """A breached --gate regression bar must mark the MLflow run FAILED."""
    from speechai.cli import main as cli_main
    from speechai.core.config import Settings
    from speechai.eval.metrics import EvaluationReport, UtteranceResult

    # wer mean 0.2 > wer_tolerance 0.05 -> gate fails, exit 1, run marked FAILED.
    report = EvaluationReport(
        engine="fake-stt",
        dataset="calls",
        results=[
            UtteranceResult(
                audio="a.wav", reference="r", hypothesis="h",
                wer=0.2, cer=0.1, audio_duration=1.0, latency_seconds=0.4, rtf=0.4,
            )
        ],
    )

    monkeypatch.setattr(cli_main, "run_from_manifest", lambda engine, manifest, language=None: report)
    monkeypatch.setattr(cli_main, "build_stt_engine", lambda settings: object())
    monkeypatch.setattr(
        Settings,
        "load",
        staticmethod(
            lambda cls=None: Settings(
                tracking={"enabled": True, "tracking_uri": "http://mlflow.test"},
                eval={"report_dir": str(tmp_path), "default_wer_tolerance": 0.05},
            )
        ),
    )

    exit_code = cli_main.main(
        ["evaluate", "data/manifest.jsonl", "--gate", "--report", str(tmp_path / "r.json")]
    )
    assert exit_code == 1
    assert fake_mlflow.metrics["wer_mean"][0] == 0.2  # still recorded before the gate ran
    assert fake_mlflow.finished == "FAILED"


def test_cli_evaluate_respects_no_mlflow_flag(monkeypatch, fake_mlflow, tmp_path) -> None:
    """`--no-mlflow` must disable tracking even when the config enables it."""
    from speechai.cli import main as cli_main
    from speechai.core.config import Settings
    from speechai.eval.metrics import EvaluationReport

    monkeypatch.setattr(cli_main, "run_from_manifest", lambda engine, manifest, language=None: EvaluationReport(engine="fake-stt", dataset="calls"))
    monkeypatch.setattr(cli_main, "build_stt_engine", lambda settings: object())
    monkeypatch.setattr(
        Settings,
        "load",
        staticmethod(
            lambda cls=None: Settings(
                tracking={"enabled": True, "tracking_uri": "http://mlflow.test"},
                eval={"report_dir": str(tmp_path)},
            )
        ),
    )

    exit_code = cli_main.main(
        ["evaluate", "data/manifest.jsonl", "--no-mlflow", "--report", str(tmp_path / "r.json")]
    )
    assert exit_code == 0
    assert fake_mlflow.finished is None  # nothing was recorded


def test_evaluate_skips_tracking_when_disabled(settings, tmp_path) -> None:
    from speechai.cli.main import _track_evaluation
    from speechai.eval.metrics import EvaluationReport

    settings.tracking.enabled = False
    report = EvaluationReport(engine="fake-stt", dataset="calls")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    assert _track_evaluation(settings, report, report_path, enabled=True) is None  # no tracking
    assert report_path.is_file()


def test_tracking_config_defaults() -> None:
    settings = Settings()
    assert settings.tracking.enabled is False
    assert settings.tracking.provider == "mlflow"
    assert settings.tracking.tracking_uri == ""
    assert settings.tracking.experiment_name == "speechai"


def test_env_uri_not_set(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracker = ExperimentTracker(enabled=True, tracking_uri=None, experiment_name="t")
    assert tracker.enabled is False
