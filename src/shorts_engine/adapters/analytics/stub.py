"""Stub analytics adapter for testing."""

import random
from datetime import datetime, timedelta

from shorts_engine.adapters.analytics.base import AnalyticsAdapter, MetricsSnapshot
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class StubAnalyticsAdapter(AnalyticsAdapter):
    """Stub adapter that returns simulated analytics data."""

    def __init__(self, platform: Platform = Platform.YOUTUBE) -> None:
        self._platform = platform

    @property
    def platform(self) -> Platform:
        return self._platform

    async def fetch_metrics(self, platform_video_id: str) -> MetricsSnapshot:
        """Return simulated metrics."""
        logger.info(
            "stub_fetch_metrics",
            platform=self.platform,
            platform_video_id=platform_video_id,
        )

        views = random.randint(100, 10000)
        likes = int(views * random.uniform(0.02, 0.15))
        comments = int(views * random.uniform(0.001, 0.02))
        shares = int(views * random.uniform(0.005, 0.05))

        return MetricsSnapshot(
            platform=self.platform,
            platform_video_id=platform_video_id,
            fetched_at=datetime.now(),
            views=views,
            likes=likes,
            comments_count=comments,
            shares=shares,
            watch_time_seconds=int(views * random.uniform(20, 50)),
            avg_view_duration_seconds=random.uniform(15, 45),
            engagement_rate=(likes + comments + shares) / views if views > 0 else 0,
            raw_data={"source": "stub", "platform_video_id": platform_video_id},
        )

    async def fetch_historical_metrics(
        self,
        platform_video_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[MetricsSnapshot]:
        """Return simulated historical metrics."""
        logger.info(
            "stub_fetch_historical_metrics",
            platform=self.platform,
            platform_video_id=platform_video_id,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        snapshots = []
        current_date = start_date
        cumulative_views = 0

        while current_date <= end_date:
            # Simulate viral decay pattern
            days_since_start = (current_date - start_date).days
            decay_factor = 0.7**days_since_start
            daily_views = int(random.randint(500, 5000) * decay_factor)
            cumulative_views += daily_views

            likes = int(cumulative_views * random.uniform(0.02, 0.15))
            comments = int(cumulative_views * random.uniform(0.001, 0.02))
            shares = int(cumulative_views * random.uniform(0.005, 0.05))

            snapshots.append(
                MetricsSnapshot(
                    platform=self.platform,
                    platform_video_id=platform_video_id,
                    fetched_at=current_date,
                    views=cumulative_views,
                    likes=likes,
                    comments_count=comments,
                    shares=shares,
                    watch_time_seconds=int(cumulative_views * random.uniform(20, 50)),
                    avg_view_duration_seconds=random.uniform(15, 45),
                    engagement_rate=(
                        (likes + comments + shares) / cumulative_views
                        if cumulative_views > 0
                        else 0
                    ),
                    raw_data={"source": "stub", "date": current_date.isoformat()},
                )
            )

            current_date += timedelta(days=1)

        return snapshots

    async def health_check(self) -> bool:
        """Stub adapter is always healthy."""
        return True
