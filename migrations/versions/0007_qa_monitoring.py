"""Add QA gates and operational monitoring.

Revision ID: 0007
Revises: 0006
Create Date: 2026-01-26

Adds:
- qa_status, qa_attempts, last_qa_error to video_jobs
- qa_results table for QA check history
- pipeline_metrics table for time-series metrics

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add QA fields to video_jobs
    op.add_column(
        "video_jobs",
        sa.Column("qa_status", sa.String(50), nullable=True, index=True),
    )
    op.add_column(
        "video_jobs",
        sa.Column("qa_attempts", sa.Integer, server_default="0", nullable=False),
    )
    op.add_column(
        "video_jobs",
        sa.Column("last_qa_error", sa.Text, nullable=True),
    )
    op.create_index("ix_video_jobs_qa_status", "video_jobs", ["qa_status"])

    # Create qa_results table for QA check history
    op.create_table(
        "qa_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_jobs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("check_type", sa.String(50), nullable=False),  # plan_qa, render_qa
        sa.Column("stage", sa.String(50), nullable=False),  # post_planning, post_render
        sa.Column("attempt_number", sa.Integer, nullable=False),
        sa.Column("passed", sa.Boolean, nullable=False),
        # Scores
        sa.Column("hook_clarity_score", sa.Float, nullable=True),
        sa.Column("coherence_score", sa.Float, nullable=True),
        sa.Column("uniqueness_score", sa.Float, nullable=True),
        sa.Column("policy_passed", sa.Boolean, nullable=True),
        # Details
        sa.Column("policy_violations", postgresql.JSONB, nullable=True),
        sa.Column("feedback", sa.Text, nullable=True),
        sa.Column("similar_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("similarity_score", sa.Float, nullable=True),
        # Raw response
        sa.Column("raw_response", postgresql.JSONB, nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_qa_results_check_type", "qa_results", ["check_type"])
    op.create_index("ix_qa_results_passed", "qa_results", ["passed"])

    # Create pipeline_metrics table for time-series metrics
    op.create_table(
        "pipeline_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("metric_name", sa.String(100), nullable=False, index=True),
        sa.Column("metric_type", sa.String(20), nullable=False),  # counter, gauge, histogram
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("labels", postgresql.JSONB, nullable=True),  # e.g., {stage: "planning", status: "success"}
        sa.Column(
            "video_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_jobs.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            index=True,
        ),
    )
    op.create_index("ix_pipeline_metrics_metric_name", "pipeline_metrics", ["metric_name"])
    op.create_index(
        "ix_pipeline_metrics_recorded_at",
        "pipeline_metrics",
        ["recorded_at"],
    )
    # Composite index for dashboard queries
    op.create_index(
        "ix_pipeline_metrics_name_time",
        "pipeline_metrics",
        ["metric_name", "recorded_at"],
    )


def downgrade() -> None:
    # Drop pipeline_metrics
    op.drop_index("ix_pipeline_metrics_name_time", table_name="pipeline_metrics")
    op.drop_index("ix_pipeline_metrics_recorded_at", table_name="pipeline_metrics")
    op.drop_index("ix_pipeline_metrics_metric_name", table_name="pipeline_metrics")
    op.drop_table("pipeline_metrics")

    # Drop qa_results
    op.drop_index("ix_qa_results_passed", table_name="qa_results")
    op.drop_index("ix_qa_results_check_type", table_name="qa_results")
    op.drop_table("qa_results")

    # Remove video_jobs QA columns
    op.drop_index("ix_video_jobs_qa_status", table_name="video_jobs")
    op.drop_column("video_jobs", "last_qa_error")
    op.drop_column("video_jobs", "qa_attempts")
    op.drop_column("video_jobs", "qa_status")
