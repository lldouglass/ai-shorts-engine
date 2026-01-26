"""Analytics ingestion tables for metrics, comments, and recipe features

Revision ID: 0004
Revises: 0003
Create Date: 2025-01-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Video metrics table - time-windowed performance snapshots
    op.create_table(
        "video_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("publish_job_id", sa.UUID(), nullable=False),
        sa.Column("window_type", sa.String(20), nullable=False),  # 1h, 6h, 24h, 72h, 7d, lifetime
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        # Core metrics
        sa.Column("views", sa.BigInteger(), server_default="0"),
        sa.Column("likes", sa.BigInteger(), server_default="0"),
        sa.Column("dislikes", sa.BigInteger(), server_default="0"),
        sa.Column("comments_count", sa.BigInteger(), server_default="0"),
        sa.Column("shares", sa.BigInteger(), server_default="0"),
        sa.Column("watch_time_minutes", sa.BigInteger(), server_default="0"),
        sa.Column("avg_view_duration_seconds", sa.Float(), nullable=True),
        sa.Column("avg_view_percentage", sa.Float(), nullable=True),
        # Engagement
        sa.Column("impressions", sa.BigInteger(), nullable=True),
        sa.Column("click_through_rate", sa.Float(), nullable=True),
        sa.Column("engagement_rate", sa.Float(), nullable=True),
        # Computed scores
        sa.Column("reward_score", sa.Float(), nullable=True),
        # Raw data
        sa.Column("raw_data", JSONB(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["publish_job_id"], ["publish_jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("publish_job_id", "window_type", "window_start", name="uq_video_metrics_window"),
    )
    op.create_index("ix_video_metrics_publish_job_id", "video_metrics", ["publish_job_id"])
    op.create_index("ix_video_metrics_window_type", "video_metrics", ["window_type"])
    op.create_index("ix_video_metrics_fetched_at", "video_metrics", ["fetched_at"])

    # Video comments table - top-level comments from platforms
    op.create_table(
        "video_comments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("publish_job_id", sa.UUID(), nullable=False),
        sa.Column("platform_comment_id", sa.String(255), nullable=False),
        sa.Column("author_channel_id", sa.String(255), nullable=True),
        sa.Column("author_display_name", sa.String(255), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("like_count", sa.Integer(), server_default="0"),
        sa.Column("reply_count", sa.Integer(), server_default="0"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        # Sentiment analysis (computed later)
        sa.Column("sentiment", sa.String(20), nullable=True),  # positive, negative, neutral
        sa.Column("sentiment_score", sa.Float(), nullable=True),
        sa.Column("raw_data", JSONB(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["publish_job_id"], ["publish_jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("publish_job_id", "platform_comment_id", name="uq_video_comment_unique"),
    )
    op.create_index("ix_video_comments_publish_job_id", "video_comments", ["publish_job_id"])
    op.create_index("ix_video_comments_published_at", "video_comments", ["published_at"])

    # Video recipe features table - production features for ML correlation
    op.create_table(
        "video_recipe_features",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("video_job_id", sa.UUID(), nullable=False),
        # Hook analysis
        sa.Column("hook_type", sa.String(50), nullable=True),  # question, statement, shock, etc.
        sa.Column("hook_duration_seconds", sa.Float(), nullable=True),
        # Structure
        sa.Column("scene_count", sa.Integer(), nullable=True),
        sa.Column("total_duration_seconds", sa.Float(), nullable=True),
        sa.Column("avg_scene_duration_seconds", sa.Float(), nullable=True),
        # Captions/text
        sa.Column("caption_density", sa.Float(), nullable=True),  # words per second
        sa.Column("total_word_count", sa.Integer(), nullable=True),
        # Narration
        sa.Column("has_voiceover", sa.Boolean(), server_default="false"),
        sa.Column("narration_wpm", sa.Float(), nullable=True),  # words per minute
        sa.Column("voice_id", sa.String(100), nullable=True),
        # Music
        sa.Column("has_background_music", sa.Boolean(), server_default="false"),
        sa.Column("music_genre", sa.String(50), nullable=True),
        # Style
        sa.Column("style_preset", sa.String(100), nullable=True),
        sa.Column("aspect_ratio", sa.String(20), nullable=True),
        # Raw feature vector for ML
        sa.Column("feature_vector", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["video_job_id"], ["video_jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("video_job_id", name="uq_video_recipe_features_job"),
    )
    op.create_index("ix_video_recipe_features_video_job_id", "video_recipe_features", ["video_job_id"])
    op.create_index("ix_video_recipe_features_style_preset", "video_recipe_features", ["style_preset"])


def downgrade() -> None:
    op.drop_table("video_recipe_features")
    op.drop_table("video_comments")
    op.drop_table("video_metrics")
