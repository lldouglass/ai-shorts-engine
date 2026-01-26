"""Add learning loop tables for recipe tracking and experiments.

Revision ID: 0005
Revises: 0004
Create Date: 2025-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create recipes table - canonical recipe definitions
    op.create_table(
        "recipes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        # Recipe components
        sa.Column("preset", sa.String(100), nullable=False),
        sa.Column("hook_type", sa.String(50), nullable=False),
        sa.Column("scene_count", sa.Integer, nullable=False),
        sa.Column("narration_wpm_bucket", sa.String(20), nullable=False),  # slow, medium, fast
        sa.Column("caption_density_bucket", sa.String(20), nullable=False),  # sparse, medium, dense
        sa.Column("ending_type", sa.String(50), nullable=False),  # cliffhanger, resolve
        # Recipe hash for deduplication
        sa.Column("recipe_hash", sa.String(64), nullable=False, unique=True),
        # Performance aggregates (updated periodically)
        sa.Column("times_used", sa.Integer, server_default="0"),
        sa.Column("avg_reward_score", sa.Float, nullable=True),
        sa.Column("best_reward_score", sa.Float, nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        # Metadata
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index("ix_recipes_project_avg_reward", "recipes", ["project_id", "avg_reward_score"])

    # Create experiments table - tracks A/B test experiments
    op.create_table(
        "experiments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        # Experiment type
        sa.Column("variable_tested", sa.String(50), nullable=False),  # preset, hook_type, scene_count, etc.
        sa.Column("baseline_recipe_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True),
        sa.Column("baseline_value", sa.String(100), nullable=False),
        sa.Column("variant_value", sa.String(100), nullable=False),
        # Status
        sa.Column("status", sa.String(50), server_default="running"),  # running, completed, inconclusive
        # Results
        sa.Column("baseline_video_count", sa.Integer, server_default="0"),
        sa.Column("variant_video_count", sa.Integer, server_default="0"),
        sa.Column("baseline_avg_reward", sa.Float, nullable=True),
        sa.Column("variant_avg_reward", sa.Float, nullable=True),
        sa.Column("winner", sa.String(20), nullable=True),  # baseline, variant, inconclusive
        sa.Column("confidence", sa.Float, nullable=True),  # Statistical confidence
        # Timestamps
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_experiments_project_status", "experiments", ["project_id", "status"])

    # Create planned_batches table - tracks nightly batch planning
    op.create_table(
        "planned_batches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        # Batch info
        sa.Column("batch_date", sa.Date, nullable=False),
        sa.Column("total_jobs", sa.Integer, nullable=False),
        sa.Column("exploit_count", sa.Integer, nullable=False),
        sa.Column("explore_count", sa.Integer, nullable=False),
        # Status
        sa.Column("status", sa.String(50), server_default="planned"),  # planned, running, completed, failed
        sa.Column("jobs_created", sa.Integer, server_default="0"),
        sa.Column("jobs_completed", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_planned_batches_project_date", "planned_batches", ["project_id", "batch_date"], unique=True)

    # Add recipe tracking to video_jobs
    op.add_column("video_jobs", sa.Column("recipe_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("video_jobs", sa.Column("experiment_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("video_jobs", sa.Column("generation_mode", sa.String(20), nullable=True))  # exploit, explore, manual
    op.add_column("video_jobs", sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("video_jobs", sa.Column("topic_hash", sa.String(64), nullable=True))  # For deduplication

    op.create_foreign_key(
        "fk_video_jobs_recipe",
        "video_jobs",
        "recipes",
        ["recipe_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_video_jobs_experiment",
        "video_jobs",
        "experiments",
        ["experiment_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_video_jobs_batch",
        "video_jobs",
        "planned_batches",
        ["batch_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_video_jobs_recipe_topic", "video_jobs", ["recipe_id", "topic_hash"])

    # Add ending_type to video_recipe_features
    op.add_column("video_recipe_features", sa.Column("ending_type", sa.String(50), nullable=True))


def downgrade() -> None:
    # Remove columns from video_recipe_features
    op.drop_column("video_recipe_features", "ending_type")

    # Remove foreign keys and columns from video_jobs
    op.drop_constraint("fk_video_jobs_batch", "video_jobs", type_="foreignkey")
    op.drop_constraint("fk_video_jobs_experiment", "video_jobs", type_="foreignkey")
    op.drop_constraint("fk_video_jobs_recipe", "video_jobs", type_="foreignkey")
    op.drop_index("ix_video_jobs_recipe_topic", table_name="video_jobs")
    op.drop_column("video_jobs", "topic_hash")
    op.drop_column("video_jobs", "batch_id")
    op.drop_column("video_jobs", "generation_mode")
    op.drop_column("video_jobs", "experiment_id")
    op.drop_column("video_jobs", "recipe_id")

    # Drop tables
    op.drop_index("ix_planned_batches_project_date", table_name="planned_batches")
    op.drop_table("planned_batches")

    op.drop_index("ix_experiments_project_status", table_name="experiments")
    op.drop_table("experiments")

    op.drop_index("ix_recipes_project_avg_reward", table_name="recipes")
    op.drop_table("recipes")
