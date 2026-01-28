"""YouTube Analytics adapter using Analytics and Data APIs."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx

from shorts_engine.adapters.analytics.base import AnalyticsAdapter, MetricsSnapshot
from shorts_engine.adapters.publisher.youtube_oauth import OAuthError, refresh_access_token
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger
from shorts_engine.services.accounts import (
    get_account_state,
    mark_account_revoked,
    update_account_tokens,
)

logger = get_logger(__name__)

YOUTUBE_ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
YOUTUBE_DATA_URL = "https://www.googleapis.com/youtube/v3/videos"


@dataclass
class WindowedMetrics:
    """Metrics for a specific time window."""

    window_type: str  # "1h", "6h", "24h", "72h", "7d", "lifetime"
    window_start: datetime
    window_end: datetime
    metrics: MetricsSnapshot


class YouTubeAnalyticsAdapter(AnalyticsAdapter):
    """Fetches analytics from YouTube Analytics API.

    Uses both:
    - YouTube Data API (videos.list) for real-time statistics
    - YouTube Analytics API (reports.query) for historical/windowed metrics
    """

    def __init__(self, account_id: UUID) -> None:
        """Initialize the adapter.

        Args:
            account_id: The platform account UUID to use for authentication.
        """
        self.account_id = account_id
        self._platform = Platform.YOUTUBE
        self._client: httpx.AsyncClient | None = None

    @property
    def platform(self) -> Platform:
        return self._platform

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing if needed.

        Returns:
            Valid access token.

        Raises:
            OAuthError: If token refresh fails.
        """
        with get_session_context() as session:
            account_state = get_account_state(session, self.account_id)

            # Check if token needs refresh (5 min buffer)
            if account_state.token_expires_at and account_state.token_expires_at < datetime.now(
                UTC
            ) + timedelta(minutes=5):
                logger.debug("youtube_token_refresh", account_id=str(self.account_id))
                try:
                    token_data = refresh_access_token(account_state.refresh_token)
                    new_expires = datetime.now(UTC) + timedelta(seconds=token_data["expires_in"])
                    update_account_tokens(
                        session, self.account_id, token_data["access_token"], new_expires
                    )
                    return str(token_data["access_token"])
                except OAuthError as e:
                    if "invalid_grant" in str(e):
                        mark_account_revoked(session, self.account_id, str(e))
                    raise

            return account_state.access_token

    async def fetch_metrics(self, platform_video_id: str) -> MetricsSnapshot:
        """Fetch current real-time metrics using YouTube Data API.

        Args:
            platform_video_id: The YouTube video ID.

        Returns:
            MetricsSnapshot with current stats.
        """
        access_token = await self._get_access_token()
        client = await self._get_client()

        response = await client.get(
            YOUTUBE_DATA_URL,
            params={
                "part": "statistics",
                "id": platform_video_id,
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code != 200:
            logger.error(
                "youtube_data_api_error",
                status=response.status_code,
                body=response.text[:500],
            )
            raise RuntimeError(f"YouTube Data API error: {response.status_code}")

        data = response.json()
        items = data.get("items", [])

        if not items:
            raise ValueError(f"Video not found: {platform_video_id}")

        stats = items[0].get("statistics", {})

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))

        return MetricsSnapshot(
            platform=self.platform,
            platform_video_id=platform_video_id,
            fetched_at=datetime.now(UTC),
            views=views,
            likes=likes,
            comments_count=comments,
            shares=0,  # Not available in Data API
            engagement_rate=(likes + comments) / views if views > 0 else 0,
            raw_data=stats,
        )

    async def fetch_analytics_window(
        self,
        platform_video_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> MetricsSnapshot:
        """Fetch analytics for a specific time window using Analytics API.

        Note: YouTube Analytics API has daily granularity, so sub-day windows
        will use the same data. For accurate sub-day metrics, use fetch_metrics().

        Args:
            platform_video_id: The YouTube video ID.
            start_date: Start of the date range.
            end_date: End of the date range.

        Returns:
            MetricsSnapshot for the window.
        """
        access_token = await self._get_access_token()
        client = await self._get_client()

        response = await client.get(
            YOUTUBE_ANALYTICS_URL,
            params={
                "ids": "channel==MINE",
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "metrics": "views,likes,dislikes,comments,shares,estimatedMinutesWatched,averageViewDuration,averageViewPercentage",
                "filters": f"video=={platform_video_id}",
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code == 403:
            error_data = response.json()
            if "quotaExceeded" in str(error_data):
                logger.error("youtube_analytics_quota_exceeded")
                raise RuntimeError("YouTube Analytics API quota exceeded")
            logger.error(
                "youtube_analytics_api_forbidden",
                body=response.text[:500],
            )
            raise RuntimeError(f"YouTube Analytics API forbidden: {response.status_code}")

        if response.status_code != 200:
            logger.error(
                "youtube_analytics_api_error",
                status=response.status_code,
                body=response.text[:500],
            )
            raise RuntimeError(f"YouTube Analytics API error: {response.status_code}")

        data = response.json()
        rows = data.get("rows", [])
        row = rows[0] if rows else [0] * 8

        # Column order matches metrics param order
        views, likes, dislikes, comments, shares, watch_minutes, avg_duration, avg_percentage = row

        return MetricsSnapshot(
            platform=self.platform,
            platform_video_id=platform_video_id,
            fetched_at=datetime.now(UTC),
            views=int(views),
            likes=int(likes),
            comments_count=int(comments),
            shares=int(shares),
            watch_time_seconds=int(watch_minutes * 60),
            avg_view_duration_seconds=float(avg_duration),
            engagement_rate=(likes + comments + shares) / views if views > 0 else 0,
            raw_data={
                "dislikes": dislikes,
                "avg_view_percentage": avg_percentage,
                "watch_minutes": watch_minutes,
            },
        )

    async def fetch_windowed_metrics(
        self,
        platform_video_id: str,
        publish_time: datetime,
    ) -> list[WindowedMetrics]:
        """Fetch metrics for all standard windows since publish.

        Windows: 1h, 6h, 24h, 72h, 7d (only completed windows are returned).

        Args:
            platform_video_id: The YouTube video ID.
            publish_time: When the video was published.

        Returns:
            List of WindowedMetrics for completed windows.
        """
        now = datetime.now(UTC)
        windows = []

        # Define window types relative to publish time
        window_definitions = [
            ("1h", timedelta(hours=1)),
            ("6h", timedelta(hours=6)),
            ("24h", timedelta(hours=24)),
            ("72h", timedelta(hours=72)),
            ("7d", timedelta(days=7)),
        ]

        for window_type, delta in window_definitions:
            window_end = publish_time + delta
            if window_end > now:
                continue  # Window not complete yet

            try:
                metrics = await self.fetch_analytics_window(
                    platform_video_id,
                    start_date=publish_time,
                    end_date=window_end,
                )

                windows.append(
                    WindowedMetrics(
                        window_type=window_type,
                        window_start=publish_time,
                        window_end=window_end,
                        metrics=metrics,
                    )
                )
            except Exception as e:
                logger.warning(
                    "youtube_window_metrics_error",
                    video_id=platform_video_id,
                    window=window_type,
                    error=str(e),
                )
                # Continue with other windows

            # Rate limit: pause between requests
            await asyncio.sleep(0.5)

        return windows

    async def fetch_historical_metrics(
        self,
        platform_video_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[MetricsSnapshot]:
        """Fetch daily metrics for a date range.

        Args:
            platform_video_id: The YouTube video ID.
            start_date: Start of the date range.
            end_date: End of the date range.

        Returns:
            List of MetricsSnapshot for each day.
        """
        snapshots = []
        current = start_date

        while current <= end_date:
            next_day = current + timedelta(days=1)
            try:
                metrics = await self.fetch_analytics_window(platform_video_id, current, next_day)
                snapshots.append(metrics)
            except Exception as e:
                logger.warning(
                    "youtube_historical_metrics_error",
                    video_id=platform_video_id,
                    date=current.isoformat(),
                    error=str(e),
                )

            current = next_day

            # Rate limit: 1 request per second
            await asyncio.sleep(1)

        return snapshots

    async def health_check(self) -> bool:
        """Check if YouTube Analytics API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            access_token = await self._get_access_token()
            client = await self._get_client()

            # Use a minimal query to check access
            yesterday = datetime.now(UTC) - timedelta(days=1)
            response = await client.get(
                YOUTUBE_ANALYTICS_URL,
                params={
                    "ids": "channel==MINE",
                    "startDate": yesterday.strftime("%Y-%m-%d"),
                    "endDate": yesterday.strftime("%Y-%m-%d"),
                    "metrics": "views",
                },
                headers={"Authorization": f"Bearer {access_token}"},
            )

            return response.status_code == 200
        except Exception as e:
            logger.warning("youtube_analytics_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
