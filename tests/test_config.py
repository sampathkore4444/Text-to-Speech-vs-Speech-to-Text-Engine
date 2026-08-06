"""Configuration loading tests."""

from __future__ import annotations

from speechai.core.config import Settings, _apply_env_overlay


def test_defaults() -> None:
    settings = Settings()
    assert settings.queue.backend == "memory"
    assert settings.redaction.enabled is True
    assert settings.redaction.mode == "mask"
    assert settings.stt.model_size == "base"


def test_env_overlay_numbers(monkeypatch) -> None:
    monkeypatch.setenv("SPEECHAI_API__PORT", "9000")
    settings = Settings.load()
    assert settings.api.port == 9000


def test_env_overlay_bool_and_string(monkeypatch) -> None:
    monkeypatch.setenv("SPEECHAI_REDACTION__ENABLED", "false")
    monkeypatch.setenv("SPEECHAI_STT__LANGUAGE", "en")
    settings = Settings.load()
    assert settings.redaction.enabled is False
    assert settings.stt.language == "en"


def test_env_overlay_empty_value_uses_default(monkeypatch) -> None:
    """docker compose injects SPEECHAI_API__API_KEY="" when the host variable is
    unset; an empty override must fall back to the default, not validate as None."""
    monkeypatch.setenv("SPEECHAI_API__API_KEY", "")
    monkeypatch.setenv("SPEECHAI_QUEUE__BACKEND", "")
    settings = Settings.load()
    assert settings.api.api_key == ""
    assert settings.queue.backend == "memory"


def test_env_overlay_explicit_null(monkeypatch) -> None:
    """Explicit 'null' overrides still null out nullable fields."""
    monkeypatch.setenv("SPEECHAI_STT__LANGUAGE", "null")
    settings = Settings.load()
    assert settings.stt.language is None


def test_env_overlay_nested_payload() -> None:
    payload = _apply_env_overlay({})
    assert isinstance(payload, dict)


def test_ensure_dirs(tmp_path) -> None:
    settings = Settings(storage={"data_dir": str(tmp_path / "data")})
    settings.ensure_dirs()
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "models" / "voices").is_dir()
