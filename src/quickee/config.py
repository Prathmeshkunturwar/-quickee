"""Centralized config loaded from environment.

Why pydantic-settings: validated types + fail-fast on missing required keys.
A missing GEMINI_API_KEY explodes at app startup, not at the first request.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(..., description="Google AI Studio API key")
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "models/gemini-embedding-001"
    # Matryoshka-shrunk embedding dim. 768 keeps storage tiny while retaining most quality.
    gemini_embed_dim: int = 768

    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    chroma_catalog_collection: str = "quickee_catalog"
    chroma_cache_collection: str = "quickee_prompt_cache"

    semantic_cache_threshold: float = 0.93

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def data_raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
