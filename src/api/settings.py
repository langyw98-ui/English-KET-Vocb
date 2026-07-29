from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_PATH: str = "ket_partner.db"
    CSV_PATH: str | None = None
    AUTH_MODE: Literal["disabled", "jwt"] = "disabled"
    KID_NICKNAME: str = "宝贝"
    KID_AGE: int = 8
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    REQUEST_TIMEOUT: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
