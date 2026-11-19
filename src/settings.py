
from typing import Literal
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Support Ticket Classification API"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    database_url: str | None = None
    monitoring_database_path: str = "data/monitoring.db"
    monitoring_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_prefix="TICKET_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
