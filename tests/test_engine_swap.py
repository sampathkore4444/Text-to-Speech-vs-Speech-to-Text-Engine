"""STT engine model-swap tests: selecting a fine-tuned checkpoint via config."""

from __future__ import annotations

from speechai.core.config import Settings, STTConfig
from speechai.stt.whisper_engine import WhisperSTTEngine


def test_model_path_config_accepted() -> None:
    settings = Settings(
        stt={"model_size": "base", "model_path": "data/models/finetuned/ct2"}
    )
    assert settings.stt.model_path == "data/models/finetuned/ct2"
    assert settings.stt.model_size == "base"


def test_engine_prefers_model_path() -> None:
    engine = WhisperSTTEngine(
        STTConfig(model_size="base", model_path="data/models/finetuned/ct2")
    )
    assert engine._model_ref() == "data/models/finetuned/ct2"


def test_engine_defaults_to_model_size() -> None:
    engine = WhisperSTTEngine(STTConfig(model_size="small"))
    assert engine._model_ref() == "small"
