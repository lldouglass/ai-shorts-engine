"""SQLAlchemy ORM models."""

from datetime import datetime
from typing import Any
from uuid import UUID as PyUUID
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# =============================================================================
# Video Pipeline Models (new)
# =============================================================================


class ProjectModel(Base):
    """Project (content brand / channel) ORM model."""

    __tablename__ = "projects"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_style_preset: Mapped[str | None] = mapped_column(String(100), nullable=True)
    settings: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    video_jobs: Mapped[list["VideoJobModel"]] = relationship(
        "VideoJobModel", back_populates="project", cascade="all, delete-orphan"
    )


class VideoJobModel(Base):
    """Video job (one per intended short) ORM model."""

    __tablename__ = "video_jobs"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    idea: Mapped[str] = mapped_column(Text, nullable=False)
    style_preset: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), server_default="pending", index=True)
    stage: Mapped[str] = mapped_column(String(50), server_default="created", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    plan_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )
    # Learning loop fields
    recipe_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True
    )
    experiment_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True
    )
    generation_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)  # exploit, explore, manual
    batch_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("planned_batches.id", ondelete="SET NULL"), nullable=True
    )
    topic_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # For deduplication
    # QA fields
    qa_status: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    qa_attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    last_qa_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["ProjectModel"] = relationship("ProjectModel", back_populates="video_jobs")
    scenes: Mapped[list["SceneModel"]] = relationship(
        "SceneModel", back_populates="video_job", cascade="all, delete-orphan"
    )
    assets: Mapped[list["AssetModel"]] = relationship(
        "AssetModel", back_populates="video_job", cascade="all, delete-orphan"
    )
    recipe_features: Mapped["VideoRecipeFeaturesModel | None"] = relationship(
        "VideoRecipeFeaturesModel", back_populates="video_job", uselist=False, cascade="all, delete-orphan"
    )
    recipe: Mapped["RecipeModel | None"] = relationship(
        "RecipeModel", back_populates="video_jobs"
    )
    experiment: Mapped["ExperimentModel | None"] = relationship(
        "ExperimentModel", back_populates="video_jobs"
    )
    batch: Mapped["PlannedBatchModel | None"] = relationship(
        "PlannedBatchModel", back_populates="video_jobs"
    )
    qa_results: Mapped[list["QAResultModel"]] = relationship(
        "QAResultModel", back_populates="video_job", cascade="all, delete-orphan"
    )


class SceneModel(Base):
    """Scene (per video job, ordered) ORM model."""

    __tablename__ = "scenes"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), index=True
    )
    scene_number: Mapped[int] = mapped_column(Integer, nullable=False)
    visual_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    continuity_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_beat: Mapped[str | None] = mapped_column(String(100), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, server_default="5.0")
    status: Mapped[str] = mapped_column(String(50), server_default="pending", index=True)
    generation_attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("video_job_id", "scene_number", name="uq_scene_number"),
    )

    # Relationships
    video_job: Mapped["VideoJobModel"] = relationship("VideoJobModel", back_populates="scenes")
    assets: Mapped[list["AssetModel"]] = relationship(
        "AssetModel", back_populates="scene", cascade="all, delete-orphan"
    )
    prompts: Mapped[list["PromptModel"]] = relationship(
        "PromptModel", back_populates="scene", cascade="all, delete-orphan"
    )


class AssetModel(Base):
    """Asset (clips/audio/final) ORM model."""

    __tablename__ = "assets"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), index=True
    )
    scene_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    storage_type: Mapped[str] = mapped_column(String(50), server_default="local")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), server_default="pending", index=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    video_job: Mapped["VideoJobModel"] = relationship("VideoJobModel", back_populates="assets")
    scene: Mapped["SceneModel | None"] = relationship("SceneModel", back_populates="assets")


class PromptModel(Base):
    """Prompt (exact prompts used per scene) ORM model."""

    __tablename__ = "prompts"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    scene_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), index=True
    )
    prompt_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    version: Mapped[int] = mapped_column(Integer, server_default="1")
    is_final: Mapped[bool] = mapped_column(Boolean, server_default="false")
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    scene: Mapped["SceneModel"] = relationship("SceneModel", back_populates="prompts")


# =============================================================================
# Original Models (kept for backward compatibility)
# =============================================================================


class VideoModel(Base):
    """Video ORM model."""

    __tablename__ = "videos"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=60)
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    publish_results: Mapped[list["PublishResultModel"]] = relationship(
        "PublishResultModel", back_populates="video", cascade="all, delete-orphan"
    )


class PublishResultModel(Base):
    """Publish result ORM model."""

    __tablename__ = "publish_results"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    platform: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    platform_video_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    video: Mapped["VideoModel"] = relationship("VideoModel", back_populates="publish_results")
    metrics: Mapped[list["PerformanceMetricsModel"]] = relationship(
        "PerformanceMetricsModel", back_populates="publish_result", cascade="all, delete-orphan"
    )
    comments: Mapped[list["CommentModel"]] = relationship(
        "CommentModel", back_populates="publish_result", cascade="all, delete-orphan"
    )


class PerformanceMetricsModel(Base):
    """Performance metrics ORM model."""

    __tablename__ = "performance_metrics"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    publish_result_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("publish_results.id", ondelete="CASCADE"),
        index=True,
    )
    views: Mapped[int] = mapped_column(BigInteger, default=0)
    likes: Mapped[int] = mapped_column(BigInteger, default=0)
    comments_count: Mapped[int] = mapped_column(BigInteger, default=0)
    shares: Mapped[int] = mapped_column(BigInteger, default=0)
    watch_time_seconds: Mapped[int] = mapped_column(BigInteger, default=0)
    avg_view_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Relationships
    publish_result: Mapped["PublishResultModel"] = relationship(
        "PublishResultModel", back_populates="metrics"
    )


class CommentModel(Base):
    """Comment ORM model."""

    __tablename__ = "comments"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    publish_result_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("publish_results.id", ondelete="CASCADE"),
        index=True,
    )
    platform_comment_id: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str | None] = mapped_column(String(50), nullable=True)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("publish_result_id", "platform_comment_id", name="uq_comment_unique"),
    )

    # Relationships
    publish_result: Mapped["PublishResultModel"] = relationship(
        "PublishResultModel", back_populates="comments"
    )


class JobModel(Base):
    """Job ORM model."""

    __tablename__ = "jobs"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    job_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    input_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    output_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# =============================================================================
# Platform Accounts Models (multi-account publishing)
# =============================================================================


class PlatformAccountModel(Base):
    """Platform account (connected YouTube, TikTok, etc.) ORM model."""

    __tablename__ = "platform_accounts"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    platform: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    encrypted_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_access_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scopes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), server_default="active", index=True)
    uploads_today: Mapped[int] = mapped_column(Integer, server_default="0")
    uploads_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)  # e.g., {direct_post: true}
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("platform", "label", name="uq_platform_label"),
    )

    # Relationships
    account_projects: Mapped[list["AccountProjectModel"]] = relationship(
        "AccountProjectModel", back_populates="account", cascade="all, delete-orphan"
    )
    publish_jobs: Mapped[list["PublishJobModel"]] = relationship(
        "PublishJobModel", back_populates="account", cascade="all, delete-orphan"
    )


class AccountProjectModel(Base):
    """Account-project mapping ORM model."""

    __tablename__ = "account_projects"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    account_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("platform_accounts.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    is_default: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("account_id", "project_id", name="uq_account_project"),
    )

    # Relationships
    account: Mapped["PlatformAccountModel"] = relationship(
        "PlatformAccountModel", back_populates="account_projects"
    )
    project: Mapped["ProjectModel"] = relationship("ProjectModel")


class PublishJobModel(Base):
    """Publish job (track individual publish operations) ORM model."""

    __tablename__ = "publish_jobs"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), index=True
    )
    account_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("platform_accounts.id", ondelete="CASCADE"), index=True
    )
    platform: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), server_default="pending", index=True)
    platform_video_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    platform_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    scheduled_publish_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_publish_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    visibility: Mapped[str] = mapped_column(String(50), server_default="public")
    forced_private: Mapped[bool] = mapped_column(Boolean, server_default="false")
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    dry_run: Mapped[bool] = mapped_column(Boolean, server_default="false")
    dry_run_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    api_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    manual_publish_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)  # For NEEDS_MANUAL_PUBLISH
    share_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)  # TikTok Share Intent fallback
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    video_job: Mapped["VideoJobModel"] = relationship("VideoJobModel")
    account: Mapped["PlatformAccountModel"] = relationship(
        "PlatformAccountModel", back_populates="publish_jobs"
    )
    metrics: Mapped[list["VideoMetricsModel"]] = relationship(
        "VideoMetricsModel", back_populates="publish_job", cascade="all, delete-orphan"
    )
    video_comments: Mapped[list["VideoCommentModel"]] = relationship(
        "VideoCommentModel", back_populates="publish_job", cascade="all, delete-orphan"
    )


# =============================================================================
# Analytics Ingestion Models
# =============================================================================


class VideoMetricsModel(Base):
    """Video performance metrics by time window."""

    __tablename__ = "video_metrics"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    publish_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("publish_jobs.id", ondelete="CASCADE"), index=True
    )
    window_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 1h, 6h, 24h, 72h, 7d
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Core metrics
    views: Mapped[int] = mapped_column(BigInteger, server_default="0")
    likes: Mapped[int] = mapped_column(BigInteger, server_default="0")
    dislikes: Mapped[int] = mapped_column(BigInteger, server_default="0")
    comments_count: Mapped[int] = mapped_column(BigInteger, server_default="0")
    shares: Mapped[int] = mapped_column(BigInteger, server_default="0")
    watch_time_minutes: Mapped[int] = mapped_column(BigInteger, server_default="0")
    avg_view_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_view_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Engagement
    impressions: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    click_through_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Computed scores
    reward_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Raw data
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("publish_job_id", "window_type", "window_start", name="uq_video_metrics_window"),
    )

    # Relationships
    publish_job: Mapped["PublishJobModel"] = relationship(
        "PublishJobModel", back_populates="metrics"
    )


class VideoCommentModel(Base):
    """Video comments from platforms."""

    __tablename__ = "video_comments"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    publish_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("publish_jobs.id", ondelete="CASCADE"), index=True
    )
    platform_comment_id: Mapped[str] = mapped_column(String(255), nullable=False)
    author_channel_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    author_display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    like_count: Mapped[int] = mapped_column(Integer, server_default="0")
    reply_count: Mapped[int] = mapped_column(Integer, server_default="0")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Sentiment analysis (computed later)
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("publish_job_id", "platform_comment_id", name="uq_video_comment_unique"),
    )

    # Relationships
    publish_job: Mapped["PublishJobModel"] = relationship(
        "PublishJobModel", back_populates="video_comments"
    )


class VideoRecipeFeaturesModel(Base):
    """Video production features for ML correlation."""

    __tablename__ = "video_recipe_features"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), index=True
    )
    # Hook analysis
    hook_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    hook_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Structure
    scene_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_scene_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Captions/text
    caption_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Narration
    has_voiceover: Mapped[bool] = mapped_column(Boolean, server_default="false")
    narration_wpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    voice_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Music
    has_background_music: Mapped[bool] = mapped_column(Boolean, server_default="false")
    music_genre: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Style
    style_preset: Mapped[str | None] = mapped_column(String(100), nullable=True)
    aspect_ratio: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Ending
    ending_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # cliffhanger, resolve
    # Raw feature vector for ML
    feature_vector: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("video_job_id", name="uq_video_recipe_features_job"),
    )

    # Relationships
    video_job: Mapped["VideoJobModel"] = relationship(
        "VideoJobModel", back_populates="recipe_features"
    )


# =============================================================================
# Learning Loop Models
# =============================================================================


class RecipeModel(Base):
    """Canonical recipe definition for video generation."""

    __tablename__ = "recipes"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # Recipe components
    preset: Mapped[str] = mapped_column(String(100), nullable=False)
    hook_type: Mapped[str] = mapped_column(String(50), nullable=False)
    scene_count: Mapped[int] = mapped_column(Integer, nullable=False)
    narration_wpm_bucket: Mapped[str] = mapped_column(String(20), nullable=False)  # slow, medium, fast
    caption_density_bucket: Mapped[str] = mapped_column(String(20), nullable=False)  # sparse, medium, dense
    ending_type: Mapped[str] = mapped_column(String(50), nullable=False)  # cliffhanger, resolve
    # Recipe hash for deduplication
    recipe_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # Performance aggregates (updated periodically)
    times_used: Mapped[int] = mapped_column(Integer, server_default="0")
    avg_reward_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_reward_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    project: Mapped["ProjectModel"] = relationship("ProjectModel")
    video_jobs: Mapped[list["VideoJobModel"]] = relationship(
        "VideoJobModel", back_populates="recipe"
    )
    experiments: Mapped[list["ExperimentModel"]] = relationship(
        "ExperimentModel", back_populates="baseline_recipe", foreign_keys="ExperimentModel.baseline_recipe_id"
    )


class ExperimentModel(Base):
    """A/B test experiment tracking."""

    __tablename__ = "experiments"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Experiment type
    variable_tested: Mapped[str] = mapped_column(String(50), nullable=False)
    baseline_recipe_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True
    )
    baseline_value: Mapped[str] = mapped_column(String(100), nullable=False)
    variant_value: Mapped[str] = mapped_column(String(100), nullable=False)
    # Status
    status: Mapped[str] = mapped_column(String(50), server_default="running")
    # Results
    baseline_video_count: Mapped[int] = mapped_column(Integer, server_default="0")
    variant_video_count: Mapped[int] = mapped_column(Integer, server_default="0")
    baseline_avg_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    variant_avg_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    winner: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    project: Mapped["ProjectModel"] = relationship("ProjectModel")
    baseline_recipe: Mapped["RecipeModel | None"] = relationship(
        "RecipeModel", back_populates="experiments", foreign_keys=[baseline_recipe_id]
    )
    video_jobs: Mapped[list["VideoJobModel"]] = relationship(
        "VideoJobModel", back_populates="experiment"
    )


class PlannedBatchModel(Base):
    """Nightly batch planning record."""

    __tablename__ = "planned_batches"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # Batch info
    batch_date: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    total_jobs: Mapped[int] = mapped_column(Integer, nullable=False)
    exploit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    explore_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Status
    status: Mapped[str] = mapped_column(String(50), server_default="planned")
    jobs_created: Mapped[int] = mapped_column(Integer, server_default="0")
    jobs_completed: Mapped[int] = mapped_column(Integer, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    project: Mapped["ProjectModel"] = relationship("ProjectModel")
    video_jobs: Mapped[list["VideoJobModel"]] = relationship(
        "VideoJobModel", back_populates="batch"
    )


# =============================================================================
# QA and Monitoring Models
# =============================================================================


class QAResultModel(Base):
    """QA check result for a video job."""

    __tablename__ = "qa_results"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_job_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), index=True
    )
    check_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # plan_qa, render_qa
    stage: Mapped[str] = mapped_column(String(50), nullable=False)  # post_planning, post_render
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    # Scores
    hook_clarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coherence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    uniqueness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Details
    policy_violations: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    similar_job_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Raw response
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    video_job: Mapped["VideoJobModel"] = relationship(
        "VideoJobModel", back_populates="qa_results"
    )


class PipelineMetricModel(Base):
    """Time-series metrics for pipeline monitoring."""

    __tablename__ = "pipeline_metrics"

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_type: Mapped[str] = mapped_column(String(20), nullable=False)  # counter, gauge, histogram
    value: Mapped[float] = mapped_column(Float, nullable=False)
    labels: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    video_job_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
