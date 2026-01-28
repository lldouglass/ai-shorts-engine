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

    # Topic Provider
    topic_provider: str = Field(
        default="stub",
        description="Topic generation provider (llm, stub)",
    )

    # Autonomous Pipeline
    auto_chain_render: bool = Field(
        default=False,
        description="Automatically chain render after video generation completes",
    )
    auto_chain_publish: bool = Field(
        default=False,
        description="Automatically chain publish after render completes",
    )
    autonomous_batch_size: int = Field(
        default=5,
        description="Number of videos to generate per autonomous batch",
    )
    autonomous_enabled: bool = Field(
        default=False,
        description="Enable fully autonomous video generation loop",
    )

    # Visual Generation Mode
    visual_gen_mode: str = Field(
        default="video",
        description="Visual generation mode: 'video' for AI video clips, 'image' for AI images with motion",
    )
    image_gen_provider: str = Field(
        default="stub",
        description="Image generation provider (stub, dalle3)",
    )
    image_gen_quality: str = Field(
        default="hd",
        description="Image quality for DALL-E (standard, hd)",
    )

    # API Keys (optional, for real providers)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    luma_api_key: str | None = Field(default=None, description="Luma AI API key")
    creatomate_api_key: str | None = Field(default=None, description="Creatomate API key")
    creatomate_webhook_url: str | None = Field(
        default=None, description="Creatomate webhook URL for render completion"
    )
    elevenlabs_api_key: str | None = Field(default=None, description="ElevenLabs API key")
    instagram_access_token: str | None = Field(default=None, description="Instagram access token")

    # YouTube OAuth (for multi-account publishing)
    youtube_client_id: str | None = Field(default=None, description="YouTube OAuth client ID")
    youtube_client_secret: str | None = Field(
        default=None, description="YouTube OAuth client secret"
    )
    youtube_redirect_uri: str = Field(
        default="http://localhost:8085/callback",
        description="YouTube OAuth redirect URI",
    )

    # Instagram OAuth (via Facebook Login / Meta Graph API)
    instagram_app_id: str | None = Field(default=None, description="Instagram/Facebook App ID")
    instagram_app_secret: str | None = Field(
        default=None, description="Instagram/Facebook App Secret"
    )
    instagram_redirect_uri: str = Field(
        default="http://localhost:8085/instagram/callback",
        description="Instagram OAuth redirect URI",
    )

    # TikTok OAuth
    tiktok_client_key: str | None = Field(default=None, description="TikTok Client Key")
    tiktok_client_secret: str | None = Field(default=None, description="TikTok Client Secret")
    tiktok_redirect_uri: str = Field(
        default="http://localhost:8085/tiktok/callback",
        description="TikTok OAuth redirect URI",
    )

    # Encryption (for token storage)
    encryption_master_key: str | None = Field(
        default=None,
        description="Fernet encryption key for secure token storage",
    )

    # Publishing guardrails
    youtube_max_uploads_per_day: int = Field(
        default=50,
        description="Maximum YouTube uploads per account per day",
    )
    instagram_max_posts_per_day: int = Field(
        default=25,
        description="Maximum Instagram posts per account per day",
    )
    tiktok_max_posts_per_day: int = Field(
        default=50,
        description="Maximum TikTok posts per account per day",
    )

    # QA Configuration
    qa_enabled: bool = Field(
        default=True,
        description="Enable QA validation gates in the pipeline",
    )
    qa_hook_clarity_threshold: float = Field(
        default=0.7,
        description="Minimum hook clarity score (0.0-1.0) to pass QA",
    )
    qa_coherence_threshold: float = Field(
        default=0.7,
        description="Minimum coherence score (0.0-1.0) to pass QA",
    )
    qa_max_regeneration_attempts: int = Field(
        default=2,
        description="Maximum times to regenerate plan on QA failure",
    )
    qa_uniqueness_lookback_count: int = Field(
        default=30,
        description="Number of recent scripts to check for uniqueness",
    )
    qa_uniqueness_similarity_threshold: float = Field(
        default=0.85,
        description="Maximum Jaccard similarity allowed with recent scripts",
    )

    # Alerting
    alert_discord_webhook_url: str | None = Field(
        default=None,
        description="Discord webhook URL for alerts",
    )
    alert_email_smtp_host: str | None = Field(
        default=None,
        description="SMTP host for email alerts",
    )
    alert_email_smtp_port: int = Field(
        default=587,
        description="SMTP port for email alerts",
    )
    alert_email_from: str | None = Field(
        default=None,
        description="From address for email alerts",
    )
    alert_email_to: list[str] = Field(
        default_factory=list,
        description="Recipient addresses for email alerts",
    )
    alert_email_username: str | None = Field(
        default=None,
        description="SMTP username for email alerts",
    )
    alert_email_password: str | None = Field(
        default=None,
        description="SMTP password for email alerts",
    )
    alert_on_qa_failure: bool = Field(
        default=True,
        description="Send alerts on QA validation failures",
    )
    alert_on_pipeline_failure: bool = Field(
        default=True,
        description="Send alerts on pipeline failures",
    )

    # Publishing Control
    autopublish_enabled: bool = Field(
        default=False,
        description="Enable automatic publishing (disabled by default for safety)",
    )

    # Metrics
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection",
    )
    metrics_retention_days: int = Field(
        default=30,
        description="Number of days to retain metrics data",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience alias
settings = get_settings()
