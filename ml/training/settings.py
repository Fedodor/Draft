"""Environment configuration for the ML training scripts.

The only supported local environment file is ``ml/.env``. Process environment
variables take precedence over values from that file. Credentials are never
given code defaults and are never written to logs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ML_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ML_ROOT / ".env"

_PLACEHOLDERS = {
    "changeme",
    "change-me",
    "none",
    "null",
    "replace-with-opendota-token",
    "replace-with-stratz-token",
    "your-api-key",
    "your-opendota-key",
    "your-stratz-key",
}
_warned_opendota = False


class ConfigurationError(RuntimeError):
    """Base class for environment configuration failures."""


class MissingApiKeyError(ConfigurationError):
    """Raised when a required API key is absent."""


def load_environment(env_file: Optional[Path] = None) -> Path:
    """Load the environment file without overriding process variables.

    Args:
        env_file: Optional override, primarily for tests. Defaults to ml/.env.

    Returns:
        The path that was considered for loading.
    """
    path = Path(env_file) if env_file is not None else ENV_PATH
    if path.is_file():
        load_dotenv(dotenv_path=path, override=False, encoding="utf-8")
    return path


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'").strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in _PLACEHOLDERS or lowered.startswith("replace-with-"):
        return None
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return None
    return cleaned


def get_stratz_key() -> Optional[str]:
    """Return the STRATZ key if configured, otherwise None."""
    load_environment()
    return _clean(os.environ.get("STRATZ_API_KEY"))


def require_stratz_key() -> str:
    """Return the STRATZ key or raise a clear configuration error."""
    key = get_stratz_key()
    if key is None:
        raise MissingApiKeyError(
            "STRATZ_API_KEY is not set. Copy ml/.env.example to ml/.env "
            "and set a valid STRATZ key."
        )
    return key


def get_opendota_key() -> Optional[str]:
    """Return the optional OpenDota key and warn once when it is absent."""
    global _warned_opendota
    load_environment()
    key = _clean(os.environ.get("OPENDOTA_API_KEY"))
    if key is None and not _warned_opendota:
        logger.warning(
            "OPENDOTA_API_KEY is not set; continuing without an OpenDota API key."
        )
        _warned_opendota = True
    return key


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def get_http_timeout() -> int:
    """Request timeout in seconds."""
    load_environment()
    return _positive_int("ML_HTTP_TIMEOUT_SECONDS", 30)


def get_max_retries() -> int:
    """Transient request retry count."""
    load_environment()
    return _positive_int("ML_MAX_RETRIES", 3)


def get_log_level() -> str:
    """Configured collector log level."""
    load_environment()
    value = (os.environ.get("ML_LOG_LEVEL") or "INFO").strip().upper()
    return value or "INFO"


def describe_environment() -> Dict[str, str]:
    """Return a redacted status summary safe for logs."""
    load_environment()
    return {
        "env_file": str(ENV_PATH),
        "env_file_exists": str(ENV_PATH.is_file()),
        "STRATZ_API_KEY": "set" if _clean(os.environ.get("STRATZ_API_KEY")) else "missing",
        "OPENDOTA_API_KEY": "set" if _clean(os.environ.get("OPENDOTA_API_KEY")) else "missing",
    }
