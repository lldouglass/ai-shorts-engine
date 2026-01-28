"""Instagram Analytics adapter using Instagram Graph API Insights."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx

from shorts_engine.adapters.analytics.base import AnalyticsAdapter, MetricsSnapshot
from shorts_engine.adapters.publisher.instagram_oauth import (
    InstagramOAuthError,
    refresh_instagram_token,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger
from shorts_engine.services.accounts import (
    get_account_state,
    mark_account_revoked,
    update_account_tokens,
)

logger = get_logger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v18.0"


@dataclass
class WindowedMetrics:
    """Metrics for a specific time window."""

    window_type: str  # "1h", "6h", "24h", "72h", "7d", "lifetime"
    window_start: datetime
    window_end: datetime
    metrics: MetricsSnapshot


class InstagramAnalyticsAdapter(AnalyticsAdapter):
    """Fetches analytics from Instagram Graph API Insights.

    Uses the Instagram Media Insights endpoint for Reels metrics.
    Available metrics include plays, reach, likes, comments, shares, and saved.
    """

    def __init__(self, account_id: UUID) -> None:
        """Initialize the adapter.

        Args:
            account_id: The platform account UUID to use for authentication.
        """
        self.account_id = account_id
        self._platform = Platform.INSTAGRAM
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
            InstagramOAuthError: If token refresh fails.
        """
        with get_session_context() as session:
            account_state = get_account_state(session, self.account_id)

            # Check if token needs refresh (7 days before expiry)
            if account_state.token_expires_at and account_state.token_expires_at < datetime.now(
                UTC
            ) + timedelta(days=7):
                logger.debug("instagram_token_refresh", account_id=str(self.account_id))
                try:
                    token_data = refresh_instagram_token(account_state.access_token)
                    new_expires = datetime.now(UTC) + timedelta(
                        seconds=token_data.get("expires_in", 5184000)
                    )
                    update_account_tokens(
                        session, self.account_id, token_data["access_token"], new_expires
                    )
                    return str(token_data["access_token"])
                except InstagramOAuthError as e:
                    if "expired" in str(e).lower():
                        mark_account_revoked(session, self.account_id, str(e))
                    raise

            return account_state.access_token

    async def fetch_metrics(self, platform_video_id: str) -> MetricsSnapshot:
        """Fetch current metrics for an Instagram Reel.

        Args:
            platform_video_id: The Instagram media ID.

        Returns:
            MetricsSnapshot with current stats.
        """
        access_token = await self._get_access_token()
        client = await self._get_client()

        # Get media basic info
        media_response = await client.get(
            f"{GRAPH_API_URL}/{platform_video_id}",
            params={
                "fields": "id,media_type,like_count,comments_count,timestamp",
                "access_token": access_token,
            },
        )

        if media_response.status_code != 200:
            logger.error(
                "instagram_media_api_error",
                status=media_response.status_code,
                body=media_response.text[:500],
            )
            raise RuntimeError(f"Instagram Media API error: {media_response.status_code}")

        media_data = media_response.json()

        # Get insights
        insights_response = await client.get(
            f"{GRAPH_API_URL}/{platform_video_id}/insights",
            params={
                "metric": "plays,reach,saved,shares",
                "access_token": access_token,
            },
        )

        insights = {}
        if insights_response.status_code == 200:
            insights_data = insights_response.json()
            for item in insights_data.get("data", []):
                metric_name = item.get("name")
                values = item.get("values", [{}])
                if values:
                    insights[metric_name] = values[0].get("value", 0)

        views = insights.get("plays", 0)
        likes = media_data.get("like_count", 0)
        comments = media_data.get("comments_count", 0)
        shares = insights.get("shares", 0)
        reach = insights.get("reach", 0)
        saved = insights.get("saved", 0)

        engagement_rate = 0.0
        if views > 0:
            engagement_rate = (likes + comments + shares + saved) / views

        return MetricsSnapshot(
            platform=self.platform,
            platform_video_id=platform_video_id,
            fetched_at=datetime.now(UTC),
            views=views,
            likes=likes,
            comments_count=comments,
            shares=shares,
            impressions=reach,
            engagement_rate=engagement_rate,
            raw_data={
                "reach": reach,
                "saved": saved,
                "plays": views,
                "media_type": media_data.get("media_type"),
            },
        )

    async def fetch_windowed_metrics(
        self,
        platform_video_id: str,
        publish_time: datetime,
    ) -> list[WindowedMetrics]:
        """Fetch metrics for all standard windows since publish.

        Note: Instagram Insights API doesn't provide historical windowed data.
        We fetch current metrics and use them as the latest window data.

        Args:
            platform_video_id: The Instagram media ID.
            publish_time: When the video was published.

        Returns:
            List of WindowedMetrics (currently just lifetime metrics).
        """
        now = datetime.now(UTC)
        windows = []

        # Fetch current metrics as "lifetime"
        try:
            metrics = await self.fetch_metrics(platform_video_id)
            windows.append(
                WindowedMetrics(
                    window_type="lifetime",
                    window_start=publish_time,
                    window_end=now,
                    metrics=metrics,
                )
            )
        except Exception as e:
            logger.warning(
                "instagram_metrics_error",
                video_id=platform_video_id,
                error=str(e),
            )

        return windows

    async def fetch_historical_metrics(
        self,
        platform_video_id: str,
        _start_date: datetime,
        _end_date: datetime,
    ) -> list[MetricsSnapshot]:
        """Fetch historical metrics for a date range.

        Note: Instagram Insights API doesn't provide historical daily metrics
        like YouTube Analytics. We return the current snapshot only.

        Args:
            platform_video_id: The Instagram media ID.
            start_date: Start of the date range.
            end_date: End of the date range.

        Returns:
            List containing current MetricsSnapshot.
        """
        try:
            metrics = await self.fetch_metrics(platform_video_id)
            return [metrics]
        except Exception as e:
            logger.warning(
                "instagram_historical_metrics_error",
                video_id=platform_video_id,
                error=str(e),
            )
            return []

    async def health_check(self) -> bool:
        """Check if Instagram API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            await self._get_access_token()
            return True
        except Exception as e:
            logger.warning("instagram_analytics_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
