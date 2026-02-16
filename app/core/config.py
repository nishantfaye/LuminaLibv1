"""Application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    database_url: str = "postgresql+asyncpg://lumina:lumina@localhost:5432/luminalib"
    storage_backend: Literal["local", "s3"] = "local"
    storage_path: str = "./storage"
    s3_bucket: str = "luminalib-books"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str = ""  # e.g. http://minio:9000 for MinIO
    s3_access_key: str = ""
    s3_secret_key: str = ""
    llm_provider: Literal["mock", "llama", "openai"] = "mock"
    llm_base_url: str = "http://localhost:11434"  # Ollama endpoint
    llm_model: str = "llama3"  # Ollama model tag
    llm_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    secret_key: str = "dev-secret-key"
    access_token_expire_minutes: int = 60

    model_config = {"env_file": ".env"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
