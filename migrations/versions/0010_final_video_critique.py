"""Add critique_iteration for final video critique loop.

Revision ID: 0010
Revises: 0009
Create Date: 2026-02-03

Adds:
- critique_iteration column to video_jobs for tracking final video critique loop

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add critique_iteration column to video_jobs
    op.add_column(
        "video_jobs",
        sa.Column(
            "critique_iteration",
            sa.Integer,
            server_default="0",
            nullable=False,
        ),
    )


def downgrade() -> None:
    # Remove critique_iteration from video_jobs
    op.drop_column("video_jobs", "critique_iteration")
