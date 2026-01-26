"""Application configuration via environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://shorts:shorts@localhost:5432/shorts",
        description="PostgreSQL connection string",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )

    # API Server
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Providers
    video_gen_provider: str = Field(
        default="stub",
        description="Video generation provider (stub, luma)",
    )
    renderer_provider: str = Field(
        default="stub",
        description="Rendering provider (stub, creatomate)",
    )
    voiceover_provider: str = Field(
        default="stub",
        description="Voiceover provider (stub, elevenlabs, edge_tts)",
    )

    # Platform Publishing
    publisher_youtube_enabled: bool = Field(default=False, description="Enable YouTube publishing")
    publisher_tiktok_enabled: bool = Field(default=False, description="Enable TikTok publishing")
    publisher_instagram_enabled: bool = Field(
        default=False, description="Enable Instagram publishing"
    )

    # LLM Provider
    llm_provider: str = Field(
        default="openai",
        description="LLM provider for planning (openai, anthropic, stub)",
    )

    # API Keys (optional, for real providers)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    luma_api_key: str | None = Field(default=None, description="Luma AI API key")
    creatomate_api_key: str | None = Field(default=None, description="Creatomate API key")
    creatomate_webhook_url: str | None = Field(default=None, description="Creatomate webhook URL for render completion")
    elevenlabs_api_key: str | None = Field(default=None, description="ElevenLabs API key")
    youtube_api_key: str | None = Field(default=None, description="YouTube API key")
    tiktok_client_key: str | None = Field(default=None, description="TikTok client key")
    instagram_access_token: str | None = Field(default=None, description="Instagram access token")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience alias
settings = get_settings()
