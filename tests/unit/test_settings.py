
import pytest
from pydantic import ValidationError

from src.settings import Settings


def test_default_settings():
    settings = Settings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_log_level_accepts_debug():
    settings = Settings(log_level="DEBUG")

    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="INVALID")


def test_app_name_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TICKET_APP_NAME", "Ticket API Test")

    from src.settings import get_settings

    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.app_name == "Ticket API Test"
    finally:
        get_settings.cache_clear()


def test_app_version_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TICKET_APP_VERSION", "2.0.0")

    from src.settings import get_settings

    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.app_version == "2.0.0"
    finally:
        get_settings.cache_clear()
