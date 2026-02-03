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
        description="Video generation provider (stub, luma, veo)",
    )
    renderer_provider: str = Field(
        default="stub",
        description="Rendering provider (stub, creatomate, ffmpeg)",
    )

    # FFmpeg settings (for renderer_provider=ffmpeg)
    ffmpeg_path: str | None = Field(
        default=None,
        description="Path to FFmpeg binary (uses 'ffmpeg' from PATH if not specified)",
    )
    ffmpeg_preset: str = Field(
        default="medium",
        description="FFmpeg encoding preset (ultrafast, fast, medium, slow, veryslow)",
    )
    ffmpeg_crf: int = Field(
        default=23,
        description="FFmpeg CRF quality (0-51, lower = better quality, 23 is default)",
    )
    ffmpeg_timeout: int = Field(
        default=600,
        description="FFmpeg render timeout in seconds (default 10 minutes)",
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

    # API Keys (optional, for real providers)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for LLM provider (gpt-4o-mini, gpt-4o, etc.)",
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    luma_api_key: str | None = Field(default=None, description="Luma AI API key")
    google_api_key: str | None = Field(default=None, description="Google API key for Gemini/Veo")
    veo_model: str = Field(
        default="veo-2.0-generate-001",
        description="Veo model (veo-2.0-generate-001, veo-3.1-fast-generate-preview)",
    )
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

    # Video Generation Rate Limiting
    video_gen_rate_limit_seconds: int = Field(
        default=15,
        description="Seconds between video generation requests to avoid API rate limits",
    )

    # Story Generation
    story_target_words: int = Field(
        default=125,
        description="Target word count for generated stories",
    )
    story_max_words: int = Field(
        default=160,
        description="Maximum word count for generated stories",
    )
    story_narration_wpm: int = Field(
        default=150,
        description="Words per minute for estimating story narration duration",
    )

    # Ralph Loop (Agentic Retry for Video Generation)
    ralph_loop_enabled: bool = Field(
        default=True,
        description="Enable Ralph agentic retry loop for video generation quality",
    )
    ralph_max_iterations: int = Field(
        default=3,
        description="Maximum iterations for Ralph loop before accepting best result",
    )
    ralph_visual_coherence_threshold: float = Field(
        default=0.75,
        description="Minimum visual coherence score (0.0-1.0) for Ralph loop to pass",
    )
    ralph_style_consistency_threshold: float = Field(
        default=0.75,
        description="Minimum style consistency score (0.0-1.0) for Ralph loop to pass",
    )
    ralph_motion_coherence_threshold: float = Field(
        default=0.70,
        description="Minimum motion coherence score (0.0-1.0) for Ralph loop to pass",
    )
    ralph_temporal_consistency_threshold: float = Field(
        default=0.70,
        description="Minimum temporal consistency score (0.0-1.0) for Ralph loop to pass",
    )
    ralph_frames_per_scene: int = Field(
        default=3,
        description="Number of frames to extract per scene for video critique",
    )
    ralph_use_best_on_failure: bool = Field(
        default=True,
        description="On Ralph loop exhaustion, restore the best iteration's clips instead of failing",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience alias
settings = get_settings()
