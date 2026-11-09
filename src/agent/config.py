from functools import lru_cache
from typing import Literal
from urllib.parse import quote

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"

    postgres_user: str = "agent"
    postgres_password: str = "change_me_postgres"
    postgres_db: str = "agent"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    storage_backend: Literal["sqlite", "postgres"] = "sqlite"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None

    s3_access_key: str = "GK00000000000000000000000000000000"
    s3_secret_key: str = "dev-only-change-this-secret-before-sharing"
    s3_endpoint: str = "127.0.0.1:3900"
    s3_bucket: str = "uploads"
    s3_region: str = "garage"
    object_storage_backend: Literal["local", "s3"] = "local"

    data_dir: str = "data"
    max_upload_bytes: int = 20 * 1024 * 1024
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 180
    ocr_enabled: bool = True
    ocr_language: str = "eng"

    # Optional OpenAI-compatible embeddings endpoint. No text leaves the machine
    # unless an endpoint and API key are explicitly configured.
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    verification_model: str | None = None

    @property
    def database_url(self) -> str:
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{quote(self.redis_password, safe='')}@" if self.redis_password else ""
        host = "127.0.0.1" if self.redis_host == "localhost" else self.redis_host
        return f"redis://{auth}{host}:{self.redis_port}/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
