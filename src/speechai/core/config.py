"""Configuration loading: YAML file + ``SPEECHAI_*`` environment overlay.

Environment variables use ``__`` as a section separator, e.g.
``SPEECHAI_STT__MODEL_SIZE=small`` overrides ``stt.model_size``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

CONFIG_ENV_VAR = "SPEECHAI_CONFIG"
ENV_PREFIX = "SPEECHAI_"
_DEFAULT_CONFIG = "configs/config.yaml"


class ServiceConfig(BaseModel):
    name: str = "bank-speech-ai"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    max_upload_mb: int = 50
    timeout_seconds: int = 300


class StorageConfig(BaseModel):
    data_dir: str = "data"
    result_ttl_seconds: int = 86400
    max_results: int = 1000


class STTConfig(BaseModel):
    engine: str = "whisper"
    model_size: str = "base"
    # Optional path to a converted CTranslate2 model directory (e.g. a
    # LoRA fine-tuned checkpoint exported by `speechai-finetune`).
    # Takes precedence over `model_size`.
    model_path: str = ""
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    language: str | None = None
    vad_filter: bool = True
    min_silence_ms: int = 500
    max_segment_ms: int = 12000
    partial_interval_ms: int = 2500


class TTSConfig(BaseModel):
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"
    model_path: str = ""
    sample_rate: int = 22050
    default_speed: float = 1.0


class VADConfig(BaseModel):
    backend: Literal["webrtc", "energy", "auto"] = "auto"
    frame_ms: int = 30
    aggressiveness: int = 2
    energy_threshold_db: float = -35.0
    min_speech_ms: int = 250
    min_silence_ms: int = 400


class QueueConfig(BaseModel):
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    poll_interval_seconds: float = 1.0
    job_timeout_seconds: int = 900


_DEFAULT_REDACTION_PATTERNS = {
    "card": True,
    "account": True,
    "ifsc": True,
    "aadhaar": True,
    "pan": True,
    "ssn": True,
    "phone": True,
    "email": True,
}


class RedactionConfig(BaseModel):
    enabled: bool = True
    mode: Literal["mask", "redact", "none"] = "mask"
    mask_keep_last: int = 4
    patterns: dict[str, bool] = Field(default_factory=lambda: dict(_DEFAULT_REDACTION_PATTERNS))


class EvalConfig(BaseModel):
    default_wer_tolerance: float = 0.10
    default_rtf_tolerance: float = 0.50
    report_dir: str = "data/eval"


class Settings(BaseModel):
    """Top-level settings object. Use :meth:`Settings.load` to construct."""

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    # Derived paths (resolved lazily against the process CWD; in Docker
    # these are mounted as absolute paths).
    # ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(self.storage.data_dir)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def voices_dir(self) -> Path:
        return self.models_dir / "voices"

    @property
    def eval_report_dir(self) -> Path:
        return Path(self.eval.report_dir)

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.results_dir,
            self.models_dir,
            self.voices_dir,
            self.eval_report_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        cfg_path = Path(path) if path else Path(os.environ.get(CONFIG_ENV_VAR, _DEFAULT_CONFIG))
        payload: dict[str, Any] = {}
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
        payload = _apply_env_overlay(payload)
        return cls(**payload)


def _apply_env_overlay(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``SPEECHAI_SECTION__FIELD=value`` env vars onto the YAML payload."""
    result = json.loads(json.dumps(payload))  # deep copy
    for key, raw in sorted(os.environ.items()):
        if not key.startswith(ENV_PREFIX):
            continue
        value = raw.strip()
        if not value:
            # An empty override means "use the default" - e.g. docker compose
            # injects SPEECHAI_API__API_KEY="" when the host variable is unset.
            # Coercing "" to None would fail validation for str fields.
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("__")
        node: dict[str, Any] = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_scalar(value)
    return result


def _coerce_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
