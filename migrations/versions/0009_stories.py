"""Add stories table for story-first video generation.

Revision ID: 0009
Revises: 0008
Create Date: 2026-02-02

Adds:
- stories table for story generation and approval workflow
- story_id foreign key to video_jobs

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create stories table
    op.create_table(
        "stories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("topic", sa.Text, nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("narrative_text", sa.Text, nullable=False),
        sa.Column(
            "narrative_style", sa.String(50), nullable=False
        ),  # first-person, third-person, documentary
        sa.Column("suggested_preset", sa.String(100), nullable=False),
        sa.Column("word_count", sa.Integer, nullable=False),
        sa.Column("estimated_duration_seconds", sa.Float, nullable=False),
        sa.Column(
            "status", sa.String(50), server_default="draft", nullable=False
        ),  # draft, approved, used
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_stories_status", "stories", ["status"])

    # Add story_id column to video_jobs
    op.add_column(
        "video_jobs",
        sa.Column(
            "story_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("stories.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )


def downgrade() -> None:
    # Remove story_id from video_jobs
    op.drop_column("video_jobs", "story_id")

    # Drop stories table
    op.drop_index("ix_stories_status", table_name="stories")
    op.drop_table("stories")
