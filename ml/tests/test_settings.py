import logging

import pytest

from training import settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for name in (
        "STRATZ_API_KEY",
        "OPENDOTA_API_KEY",
        "ML_HTTP_TIMEOUT_SECONDS",
        "ML_MAX_RETRIES",
        "ML_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "ENV_PATH", tmp_path / "missing.env")
    settings._warned_opendota = False


def test_missing_stratz_key_raises():
    with pytest.raises(settings.MissingApiKeyError, match="STRATZ_API_KEY"):
        settings.require_stratz_key()


@pytest.mark.parametrize(
    "value",
    ["", "   ", "replace-with-stratz-token", "<your-key>", "CHANGEME"],
)
def test_placeholder_stratz_key_is_missing(monkeypatch, value):
    monkeypatch.setenv("STRATZ_API_KEY", value)
    with pytest.raises(settings.MissingApiKeyError):
        settings.require_stratz_key()


def test_opendota_key_is_optional_and_warns_once(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    assert settings.get_opendota_key() is None
    assert settings.get_opendota_key() is None

    warnings = [r for r in caplog.records if "OPENDOTA_API_KEY" in r.message]
    assert len(warnings) == 1


def test_process_environment_wins_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("STRATZ_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setattr(settings, "ENV_PATH", env_file)
    monkeypatch.setenv("STRATZ_API_KEY", "from-process")

    assert settings.require_stratz_key() == "from-process"


def test_env_file_is_loaded(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "STRATZ_API_KEY=stratz-from-file\nOPENDOTA_API_KEY=opendota-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "ENV_PATH", env_file)

    assert settings.require_stratz_key() == "stratz-from-file"
    assert settings.get_opendota_key() == "opendota-from-file"


def test_environment_summary_is_redacted(monkeypatch):
    monkeypatch.setenv("STRATZ_API_KEY", "secret-stratz-value")
    monkeypatch.setenv("OPENDOTA_API_KEY", "secret-opendota-value")

    summary = settings.describe_environment()

    assert summary["STRATZ_API_KEY"] == "set"
    assert summary["OPENDOTA_API_KEY"] == "set"
    assert "secret" not in str(summary)


def test_positive_integer_settings(monkeypatch):
    monkeypatch.setenv("ML_HTTP_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ML_MAX_RETRIES", "5")
    assert settings.get_http_timeout() == 45
    assert settings.get_max_retries() == 5

    monkeypatch.setenv("ML_HTTP_TIMEOUT_SECONDS", "not-a-number")
    with pytest.raises(settings.ConfigurationError):
        settings.get_http_timeout()

    monkeypatch.setenv("ML_MAX_RETRIES", "0")
    with pytest.raises(settings.ConfigurationError):
        settings.get_max_retries()
