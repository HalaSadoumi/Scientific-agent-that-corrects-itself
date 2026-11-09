from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"

    postgres_user: str = "agent"
    postgres_password: str = "change_me_postgres"
    postgres_db: str = "agent"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_host: str = "localhost"
    redis_port: int = 6379

    minio_root_user: str = "agent_minio"
    minio_root_password: str = "change_me_minio_min8chars"
    minio_endpoint: str = "localhost:9000"
    minio_bucket: str = "uploads"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
