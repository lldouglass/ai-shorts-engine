"""Add Ralph loop columns and iterations table.

Revision ID: 0008
Revises: 0007
Create Date: 2026-01-29

Adds:
- ralph_loop_enabled, ralph_current_iteration, ralph_max_iterations, ralph_status to video_jobs
- ralph_iterations table for tracking iteration history

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add Ralph loop fields to video_jobs
    op.add_column(
        "video_jobs",
        sa.Column("ralph_loop_enabled", sa.Boolean, server_default="false", nullable=False),
    )
    op.add_column(
        "video_jobs",
        sa.Column("ralph_current_iteration", sa.Integer, server_default="0", nullable=False),
    )
    op.add_column(
        "video_jobs",
        sa.Column("ralph_max_iterations", sa.Integer, server_default="3", nullable=False),
    )
    op.add_column(
        "video_jobs",
        sa.Column("ralph_status", sa.String(50), nullable=True),
    )
    op.create_index("ix_video_jobs_ralph_status", "video_jobs", ["ralph_status"])

    # Create ralph_iterations table
    op.create_table(
        "ralph_iterations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("iteration_number", sa.Integer, nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        # Generated content
        sa.Column("video_url", sa.Text, nullable=True),
        sa.Column("video_path", sa.Text, nullable=True),
        # QA Results
        sa.Column("qa_passed", sa.Boolean, nullable=True),
        sa.Column("qa_score", sa.Float, nullable=True),
        sa.Column("qa_feedback", sa.Text, nullable=True),
        sa.Column("qa_details", postgresql.JSONB, nullable=True),
        # Error handling
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("error_details", postgresql.JSONB, nullable=True),
        # Timing
        sa.Column("generation_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generation_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("qa_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("qa_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Unique constraint: one iteration number per job
        sa.UniqueConstraint("video_job_id", "iteration_number", name="uq_ralph_iteration"),
    )
    op.create_index("ix_ralph_iterations_video_job_id", "ralph_iterations", ["video_job_id"])
    op.create_index("ix_ralph_iterations_status", "ralph_iterations", ["status"])


def downgrade() -> None:
    # Drop ralph_iterations
    op.drop_index("ix_ralph_iterations_status", table_name="ralph_iterations")
    op.drop_index("ix_ralph_iterations_video_job_id", table_name="ralph_iterations")
    op.drop_table("ralph_iterations")

    # Remove video_jobs Ralph columns
    op.drop_index("ix_video_jobs_ralph_status", table_name="video_jobs")
    op.drop_column("video_jobs", "ralph_status")
    op.drop_column("video_jobs", "ralph_max_iterations")
    op.drop_column("video_jobs", "ralph_current_iteration")
    op.drop_column("video_jobs", "ralph_loop_enabled")
