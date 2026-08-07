"""Optional MLflow experiment tracking for the CLIs.

Tracking is strictly best-effort: if MLflow is not installed or no tracking
URI is configured, every method silently no-ops, so the platform, ``speechai``
and ``speechai-finetune`` keep working without any tracking dependency.

Enable it by passing ``--mlflow-tracking-uri`` (``speechai-finetune``), by
setting the ``MLFLOW_TRACKING_URI`` environment variable, or via the
``tracking`` section of ``configs/config.yaml`` (used by ``speechai evaluate``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"


class ExperimentTracker:
    """Minimal MLflow client wrapper with a no-op fallback.

    Every method is safe to call when tracking is disabled or unavailable;
    failures are logged as warnings rather than raised.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        tracking_uri: str | None,
        experiment_name: str,
        run_name: str | None = None,
    ) -> None:
        self._mlflow: Any = None
        self._enabled = False
        self._run_name = run_name
        uri = tracking_uri or os.environ.get(ENV_TRACKING_URI) or ""
        if not enabled or not uri:
            return
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "MLflow is not installed - experiment tracking disabled "
                "(install with: pip install -e '.[mlflow]')"
            )
            return
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
        except Exception as exc:  # pragma: no cover - server-specific
            logger.warning("MLflow unavailable (%s) - experiment tracking disabled", exc)
            return
        self._mlflow = mlflow
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    def start(self, tags: dict[str, str] | None = None) -> None:
        if not self._enabled:
            return
        try:
            self._mlflow.start_run(run_name=self._run_name)
            if tags:
                self._mlflow.set_tags(tags)
        except Exception as exc:  # pragma: no cover - server-specific
            logger.warning("MLflow start_run failed: %s", exc)
            self._enabled = False

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._enabled:
            return
        cleaned = {
            key: value if isinstance(value, (str, int, float, bool)) else str(value)
            for key, value in params.items()
            if value is not None
        }
        if not cleaned:
            return
        try:
            self._mlflow.log_params(cleaned)
        except Exception as exc:  # pragma: no cover - server-specific
            logger.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._enabled:
            return
        cleaned = {key: float(value) for key, value in metrics.items() if value is not None}
        if not cleaned:
            return
        try:
            self._mlflow.log_metrics(cleaned, step=step)
        except Exception as exc:  # pragma: no cover - server-specific
            logger.warning("MLflow log_metrics failed: %s", exc)

    def log_artifact(self, path: str | os.PathLike[str]) -> None:
        if not self._enabled:
            return
        local = Path(path)
        if not local.exists():
            logger.warning("MLflow artifact not found: %s", local)
            return
        try:
            if local.is_dir():
                self._mlflow.log_artifacts(str(local))
            else:
                self._mlflow.log_artifact(str(local))
        except Exception as exc:  # pragma: no cover - server-specific
            logger.warning("MLflow log_artifact failed: %s", exc)

    def end(self, status: str = "FINISHED") -> None:
        if not self._enabled:
            return
        try:
            self._mlflow.end_run(status=status)
        except TypeError:  # older mlflow: end_run() takes no arguments
            self._mlflow.end_run()
        except Exception:  # pragma: no cover - server-specific
            pass
