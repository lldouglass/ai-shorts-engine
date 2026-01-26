"""Reward score calculation for video performance evaluation.

The reward score combines multiple performance metrics into a single
normalized score that can be used for recipe comparison and optimization.

Formula:
    reward = 0.5 * avg_view_duration_score + 0.3 * views_first_6h_score + 0.2 * engagement_score

Where:
    - avg_view_duration_score: Normalized average view duration (0-1)
    - views_first_6h_score: Normalized views in first 6 hours (0-1)
    - engagement_score: (likes + comments) / views, normalized (0-1)

All terms are normalized using percentile ranking within the project to ensure
comparable scores across different content types and audience sizes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from shorts_engine.db.models import (
    PublishJobModel,
    VideoJobModel,
    VideoMetricsModel,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RewardScore:
    """Computed reward score with component breakdown."""

    total: float
    avg_view_duration_score: float
    views_6h_score: float
    engagement_score: float
    # Raw values for debugging
    raw_avg_view_duration: float | None
    raw_views_6h: int | None
    raw_engagement_rate: float | None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "components": {
                "avg_view_duration": self.avg_view_duration_score,
                "views_6h": self.views_6h_score,
                "engagement": self.engagement_score,
            },
            "raw": {
                "avg_view_duration": self.raw_avg_view_duration,
                "views_6h": self.raw_views_6h,
                "engagement_rate": self.raw_engagement_rate,
            },
        }


class RewardCalculator:
    """Calculates reward scores for videos.

    Uses project-level percentile normalization to ensure scores are
    comparable within a project regardless of absolute metric values.
    """

    # Component weights (must sum to 1.0)
    WEIGHT_AVG_VIEW_DURATION = 0.5
    WEIGHT_VIEWS_6H = 0.3
    WEIGHT_ENGAGEMENT = 0.2

    def __init__(self, session: Session, project_id: UUID):
        """Initialize calculator for a specific project.

        Args:
            session: Database session
            project_id: Project to calculate scores for
        """
        self.session = session
        self.project_id = project_id
        self._percentile_cache: dict[str, list[float]] = {}

    def _get_project_metrics(
        self,
        window_type: str = "6h",
        lookback_days: int = 30,
    ) -> list[VideoMetricsModel]:
        """Get all metrics for published videos in the project.

        Args:
            window_type: Metric window type (1h, 6h, 24h, etc.)
            lookback_days: Only consider videos published within this many days

        Returns:
            List of metrics for the project
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Get metrics for published videos in this project
        metrics = self.session.execute(
            select(VideoMetricsModel)
            .join(PublishJobModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
            .join(VideoJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .where(
                VideoJobModel.project_id == self.project_id,
                PublishJobModel.status == "published",
                PublishJobModel.actual_publish_at >= cutoff,
                VideoMetricsModel.window_type == window_type,
            )
        ).scalars().all()

        return list(metrics)

    def _build_percentile_distributions(self, lookback_days: int = 30) -> None:
        """Build percentile distributions for normalization.

        Args:
            lookback_days: Only consider videos published within this many days
        """
        metrics = self._get_project_metrics(window_type="6h", lookback_days=lookback_days)

        if not metrics:
            logger.warning("no_metrics_for_normalization", project_id=str(self.project_id))
            self._percentile_cache = {
                "avg_view_duration": [],
                "views": [],
                "engagement_rate": [],
            }
            return

        # Extract sorted values for each metric
        self._percentile_cache["avg_view_duration"] = sorted(
            [m.avg_view_duration_seconds for m in metrics if m.avg_view_duration_seconds is not None]
        )
        self._percentile_cache["views"] = sorted(
            [m.views for m in metrics if m.views is not None]
        )

        # Compute engagement rates
        engagement_rates = []
        for m in metrics:
            if m.views and m.views > 0:
                rate = (m.likes + m.comments_count) / m.views
                engagement_rates.append(rate)
        self._percentile_cache["engagement_rate"] = sorted(engagement_rates)

        logger.info(
            "percentile_distributions_built",
            project_id=str(self.project_id),
            sample_sizes={
                "avg_view_duration": len(self._percentile_cache["avg_view_duration"]),
                "views": len(self._percentile_cache["views"]),
                "engagement_rate": len(self._percentile_cache["engagement_rate"]),
            },
        )

    def _percentile_score(self, value: float | int | None, distribution_key: str) -> float:
        """Convert a raw value to a percentile score (0-1).

        Args:
            value: Raw metric value
            distribution_key: Which distribution to use for normalization

        Returns:
            Percentile score between 0 and 1
        """
        if value is None:
            return 0.5  # Default to median for missing values

        distribution = self._percentile_cache.get(distribution_key, [])

        if not distribution:
            return 0.5  # Default to median if no distribution

        # Find percentile using binary search
        import bisect
        rank = bisect.bisect_left(distribution, value)
        return rank / len(distribution) if distribution else 0.5

    def calculate(
        self,
        publish_job_id: UUID,
        lookback_days: int = 30,
    ) -> RewardScore | None:
        """Calculate reward score for a specific video.

        Args:
            publish_job_id: The publish job to calculate score for
            lookback_days: Days to look back for percentile normalization

        Returns:
            RewardScore with component breakdown, or None if no metrics
        """
        # Build distributions if not cached
        if not self._percentile_cache:
            self._build_percentile_distributions(lookback_days)

        # Get the video's 6h metrics (primary window for reward calculation)
        metrics = self.session.execute(
            select(VideoMetricsModel)
            .where(
                VideoMetricsModel.publish_job_id == publish_job_id,
                VideoMetricsModel.window_type == "6h",
            )
            .order_by(VideoMetricsModel.fetched_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if not metrics:
            logger.debug("no_metrics_for_video", publish_job_id=str(publish_job_id))
            return None

        # Calculate component scores
        avg_view_duration_score = self._percentile_score(
            metrics.avg_view_duration_seconds, "avg_view_duration"
        )
        views_score = self._percentile_score(metrics.views, "views")

        # Calculate engagement rate
        engagement_rate = None
        if metrics.views and metrics.views > 0:
            engagement_rate = (metrics.likes + metrics.comments_count) / metrics.views
        engagement_score = self._percentile_score(engagement_rate, "engagement_rate")

        # Compute weighted total
        total = (
            self.WEIGHT_AVG_VIEW_DURATION * avg_view_duration_score
            + self.WEIGHT_VIEWS_6H * views_score
            + self.WEIGHT_ENGAGEMENT * engagement_score
        )

        return RewardScore(
            total=round(total, 4),
            avg_view_duration_score=round(avg_view_duration_score, 4),
            views_6h_score=round(views_score, 4),
            engagement_score=round(engagement_score, 4),
            raw_avg_view_duration=metrics.avg_view_duration_seconds,
            raw_views_6h=metrics.views,
            raw_engagement_rate=round(engagement_rate, 4) if engagement_rate else None,
        )

    def calculate_batch(
        self,
        publish_job_ids: list[UUID],
        lookback_days: int = 30,
    ) -> dict[UUID, RewardScore]:
        """Calculate reward scores for multiple videos efficiently.

        Args:
            publish_job_ids: List of publish job IDs
            lookback_days: Days to look back for percentile normalization

        Returns:
            Dictionary mapping publish_job_id to RewardScore
        """
        if not self._percentile_cache:
            self._build_percentile_distributions(lookback_days)

        results = {}
        for job_id in publish_job_ids:
            score = self.calculate(job_id, lookback_days)
            if score:
                results[job_id] = score

        return results

    def update_stored_scores(self, lookback_days: int = 30) -> int:
        """Update reward_score column in video_metrics for all recent videos.

        Args:
            lookback_days: Days to look back

        Returns:
            Number of records updated
        """
        self._build_percentile_distributions(lookback_days)

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Get all publish jobs for this project
        publish_jobs = self.session.execute(
            select(PublishJobModel.id)
            .join(VideoJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .where(
                VideoJobModel.project_id == self.project_id,
                PublishJobModel.status == "published",
                PublishJobModel.actual_publish_at >= cutoff,
            )
        ).scalars().all()

        updated = 0
        for job_id in publish_jobs:
            score = self.calculate(job_id, lookback_days)
            if score:
                # Update all metrics for this job
                self.session.execute(
                    VideoMetricsModel.__table__.update()
                    .where(VideoMetricsModel.publish_job_id == job_id)
                    .values(reward_score=score.total)
                )
                updated += 1

        self.session.commit()
        logger.info(
            "reward_scores_updated",
            project_id=str(self.project_id),
            videos_updated=updated,
        )

        return updated
