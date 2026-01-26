"""Add multi-platform publishing support fields.

Revision ID: 0006
Revises: 0005
Create Date: 2026-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add capabilities JSONB to platform_accounts
    # Stores platform-specific capabilities like {direct_post: true} for TikTok
    op.add_column(
        "platform_accounts",
        sa.Column("capabilities", postgresql.JSONB, nullable=True),
    )

    # Add manual_publish_path to publish_jobs
    # Stores local file path when NEEDS_MANUAL_PUBLISH status is set
    op.add_column(
        "publish_jobs",
        sa.Column("manual_publish_path", sa.String(1024), nullable=True),
    )

    # Add share_url to publish_jobs
    # Stores TikTok Share Intent URL for fallback publishing
    op.add_column(
        "publish_jobs",
        sa.Column("share_url", sa.String(2048), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("publish_jobs", "share_url")
    op.drop_column("publish_jobs", "manual_publish_path")
    op.drop_column("platform_accounts", "capabilities")
