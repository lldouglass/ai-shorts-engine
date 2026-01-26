"""Video pipeline tables

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Projects table (content brands / channels)
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("default_style_preset", sa.String(100), nullable=True),
        sa.Column("settings", JSONB(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_projects_name", "projects", ["name"])
    op.create_index("ix_projects_is_active", "projects", ["is_active"])

    # Video jobs table (one per intended short)
    op.create_table(
        "video_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False, unique=True),
        sa.Column("idea", sa.Text(), nullable=False),
        sa.Column("style_preset", sa.String(100), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(50), nullable=False, server_default="created"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0"),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("plan_data", JSONB(), nullable=True),
        sa.Column("metadata_", JSONB(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_video_jobs_project_id", "video_jobs", ["project_id"])
    op.create_index("ix_video_jobs_status", "video_jobs", ["status"])
    op.create_index("ix_video_jobs_stage", "video_jobs", ["stage"])
    op.create_index("ix_video_jobs_idempotency_key", "video_jobs", ["idempotency_key"])

    # Scenes table (per job, ordered)
    op.create_table(
        "scenes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("video_job_id", sa.UUID(), nullable=False),
        sa.Column("scene_number", sa.Integer(), nullable=False),
        sa.Column("visual_prompt", sa.Text(), nullable=False),
        sa.Column("continuity_notes", sa.Text(), nullable=True),
        sa.Column("caption_beat", sa.String(100), nullable=True),
        sa.Column("duration_seconds", sa.Float(), server_default="5.0"),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("generation_attempts", sa.Integer(), server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("metadata_", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["video_job_id"], ["video_jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("video_job_id", "scene_number", name="uq_scene_number"),
    )
    op.create_index("ix_scenes_video_job_id", "scenes", ["video_job_id"])
    op.create_index("ix_scenes_status", "scenes", ["status"])

    # Assets table (clips/audio/final)
    op.create_table(
        "assets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("video_job_id", sa.UUID(), nullable=False),
        sa.Column("scene_id", sa.UUID(), nullable=True),  # NULL for job-level assets (final, audio)
        sa.Column("asset_type", sa.String(50), nullable=False),  # scene_clip, audio, final_video, thumbnail
        sa.Column("storage_type", sa.String(50), nullable=False, server_default="local"),  # local, s3, url
        sa.Column("file_path", sa.String(1024), nullable=True),
        sa.Column("url", sa.String(2048), nullable=True),
        sa.Column("external_id", sa.String(255), nullable=True),  # Provider's asset ID
        sa.Column("provider", sa.String(100), nullable=True),  # luma, runway, etc.
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("mime_type", sa.String(100), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("metadata_", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["video_job_id"], ["video_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scene_id"], ["scenes.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_assets_video_job_id", "assets", ["video_job_id"])
    op.create_index("ix_assets_scene_id", "assets", ["scene_id"])
    op.create_index("ix_assets_asset_type", "assets", ["asset_type"])
    op.create_index("ix_assets_status", "assets", ["status"])
    op.create_index("ix_assets_external_id", "assets", ["external_id"])

    # Prompts table (store exact prompts used per scene)
    op.create_table(
        "prompts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scene_id", sa.UUID(), nullable=False),
        sa.Column("prompt_type", sa.String(50), nullable=False),  # visual, audio, continuity
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("model", sa.String(100), nullable=True),  # Model used to generate/process
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("is_final", sa.Boolean(), server_default="false"),
        sa.Column("metadata_", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["scene_id"], ["scenes.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_prompts_scene_id", "prompts", ["scene_id"])
    op.create_index("ix_prompts_prompt_type", "prompts", ["prompt_type"])


def downgrade() -> None:
    op.drop_table("prompts")
    op.drop_table("assets")
    op.drop_table("scenes")
    op.drop_table("video_jobs")
    op.drop_table("projects")
