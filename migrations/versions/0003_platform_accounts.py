"""Platform accounts and publishing tables

Revision ID: 0003
Revises: 0002
Create Date: 2024-01-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Platform accounts table (connected YouTube, TikTok, etc. accounts)
    op.create_table(
        "platform_accounts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("platform", sa.String(50), nullable=False),  # youtube, tiktok, instagram
        sa.Column("label", sa.String(100), nullable=False),  # User-friendly name (e.g., "Main Channel")
        sa.Column("external_id", sa.String(255), nullable=True),  # Platform's user/channel ID
        sa.Column("external_name", sa.String(255), nullable=True),  # Channel/account name from platform
        sa.Column("encrypted_refresh_token", sa.Text(), nullable=True),  # Fernet-encrypted
        sa.Column("encrypted_access_token", sa.Text(), nullable=True),  # Fernet-encrypted (short-lived)
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scopes", sa.Text(), nullable=True),  # Comma-separated scopes
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),  # active, revoked, expired
        sa.Column("uploads_today", sa.Integer(), server_default="0"),
        sa.Column("uploads_reset_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_", JSONB(), nullable=True),  # Platform-specific data
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("platform", "label", name="uq_platform_label"),
    )
    op.create_index("ix_platform_accounts_platform", "platform_accounts", ["platform"])
    op.create_index("ix_platform_accounts_status", "platform_accounts", ["status"])
    op.create_index("ix_platform_accounts_label", "platform_accounts", ["label"])

    # Account-project mapping (which accounts are allowed for which projects)
    op.create_table(
        "account_projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("account_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default="false"),  # Default account for this project
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["account_id"], ["platform_accounts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("account_id", "project_id", name="uq_account_project"),
    )
    op.create_index("ix_account_projects_account_id", "account_projects", ["account_id"])
    op.create_index("ix_account_projects_project_id", "account_projects", ["project_id"])

    # Publish jobs table (track individual publish operations)
    op.create_table(
        "publish_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("video_job_id", sa.UUID(), nullable=False),
        sa.Column("account_id", sa.UUID(), nullable=False),
        sa.Column("platform", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("platform_video_id", sa.String(255), nullable=True),  # YouTube video ID
        sa.Column("platform_url", sa.String(2048), nullable=True),  # Public URL
        sa.Column("scheduled_publish_at", sa.DateTime(timezone=True), nullable=True),  # publishAt for scheduling
        sa.Column("actual_publish_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("visibility", sa.String(50), server_default="public"),  # public, private, unlisted
        sa.Column("forced_private", sa.Boolean(), server_default="false"),  # If API forced private
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tags", JSONB(), nullable=True),  # Array of tags
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0"),
        sa.Column("dry_run", sa.Boolean(), server_default="false"),
        sa.Column("dry_run_payload", JSONB(), nullable=True),  # What would have been sent
        sa.Column("api_response", JSONB(), nullable=True),  # Raw API response
        sa.Column("metadata_", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["video_job_id"], ["video_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["account_id"], ["platform_accounts.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_publish_jobs_video_job_id", "publish_jobs", ["video_job_id"])
    op.create_index("ix_publish_jobs_account_id", "publish_jobs", ["account_id"])
    op.create_index("ix_publish_jobs_status", "publish_jobs", ["status"])
    op.create_index("ix_publish_jobs_platform", "publish_jobs", ["platform"])


def downgrade() -> None:
    op.drop_table("publish_jobs")
    op.drop_table("account_projects")
    op.drop_table("platform_accounts")
