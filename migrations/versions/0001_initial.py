"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2024-01-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Videos table
    op.create_table(
        "videos",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("file_path", sa.String(512), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_videos_status", "videos", ["status"])
    op.create_index("ix_videos_created_at", "videos", ["created_at"])

    # Publish results table
    op.create_table(
        "publish_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("video_id", sa.UUID(), nullable=False),
        sa.Column("platform", sa.String(50), nullable=False),
        sa.Column("platform_video_id", sa.String(255), nullable=True),
        sa.Column("url", sa.String(512), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_publish_results_video_id", "publish_results", ["video_id"])
    op.create_index("ix_publish_results_platform", "publish_results", ["platform"])

    # Performance metrics table
    op.create_table(
        "performance_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("publish_result_id", sa.UUID(), nullable=False),
        sa.Column("views", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("likes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("comments_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("shares", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("watch_time_seconds", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("avg_view_duration_seconds", sa.Float(), nullable=True),
        sa.Column("engagement_rate", sa.Float(), nullable=True),
        sa.Column("raw_data", sa.JSON(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["publish_result_id"], ["publish_results.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_performance_metrics_publish_result_id",
        "performance_metrics",
        ["publish_result_id"],
    )
    op.create_index("ix_performance_metrics_fetched_at", "performance_metrics", ["fetched_at"])

    # Comments table
    op.create_table(
        "comments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("publish_result_id", sa.UUID(), nullable=False),
        sa.Column("platform_comment_id", sa.String(255), nullable=False),
        sa.Column("author", sa.String(255), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("sentiment", sa.String(50), nullable=True),
        sa.Column("likes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["publish_result_id"], ["publish_results.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("publish_result_id", "platform_comment_id", name="uq_comment_unique"),
    )
    op.create_index("ix_comments_publish_result_id", "comments", ["publish_result_id"])

    # Jobs table (for tracking job states)
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("job_type", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("input_data", sa.JSON(), nullable=True),
        sa.Column("output_data", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_celery_task_id", "jobs", ["celery_task_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("comments")
    op.drop_table("performance_metrics")
    op.drop_table("publish_results")
    op.drop_table("videos")
