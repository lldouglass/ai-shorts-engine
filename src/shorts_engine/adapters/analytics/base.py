"""Base interface for analytics ingestion adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shorts_engine.domain.enums import Platform


@dataclass
class MetricsSnapshot:
    """A snapshot of video performance metrics."""

    platform: Platform
    platform_video_id: str
    fetched_at: datetime
    views: int = 0
    likes: int = 0
    comments_count: int = 0
    shares: int = 0
    watch_time_seconds: int = 0
    avg_view_duration_seconds: float | None = None
    engagement_rate: float | None = None
    impressions: int | None = None
    click_through_rate: float | None = None
    raw_data: dict[str, Any] | None = None


class AnalyticsAdapter(ABC):
    """Abstract base class for analytics ingestion adapters.

    Implementations:
    - StubAnalyticsAdapter: Returns mock data for testing
    - YouTubeAnalyticsAdapter: Fetches from YouTube Analytics API (future)
    - TikTokAnalyticsAdapter: Fetches from TikTok Analytics API (future)
    - InstagramAnalyticsAdapter: Fetches from Instagram Insights API (future)
    """

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The platform this adapter fetches analytics from."""
        ...

    @abstractmethod
    async def fetch_metrics(self, platform_video_id: str) -> MetricsSnapshot:
        """Fetch current metrics for a video.

        Args:
            platform_video_id: The video ID on the platform

        Returns:
            MetricsSnapshot with current performance data
        """
        ...

    @abstractmethod
    async def fetch_historical_metrics(
        self,
        platform_video_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[MetricsSnapshot]:
        """Fetch historical metrics for a video over a date range.

        Args:
            platform_video_id: The video ID on the platform
            start_date: Start of the date range
            end_date: End of the date range

        Returns:
            List of MetricsSnapshot for each day in the range
        """
        ...

    async def health_check(self) -> bool:
        """Check if the analytics API is available.

        Returns:
            True if API is accessible, False otherwise
        """
        return True
