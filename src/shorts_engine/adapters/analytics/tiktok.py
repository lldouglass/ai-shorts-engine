"""TikTok Analytics adapter using TikTok API."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import httpx

from shorts_engine.adapters.analytics.base import AnalyticsAdapter, MetricsSnapshot
from shorts_engine.adapters.publisher.tiktok_oauth import (
    TikTokOAuthError,
    refresh_tiktok_token,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger
from shorts_engine.services.accounts import (
    get_account_state,
    update_account_tokens,
    mark_account_revoked,
)

logger = get_logger(__name__)

TIKTOK_API_URL = "https://open.tiktokapis.com/v2"


@dataclass
class WindowedMetrics:
    """Metrics for a specific time window."""

    window_type: str  # "1h", "6h", "24h", "72h", "7d", "lifetime"
    window_start: datetime
    window_end: datetime
    metrics: MetricsSnapshot


class TikTokAnalyticsAdapter(AnalyticsAdapter):
    """Fetches analytics from TikTok API.

    Uses the Video Query endpoint to fetch video statistics.
    Available metrics include views, likes, comments, and shares.

    Note: TikTok's analytics API has limited historical data compared to
    YouTube Analytics. Most metrics are lifetime totals.
    """

    def __init__(self, account_id: UUID) -> None:
        """Initialize the adapter.

        Args:
            account_id: The platform account UUID to use for authentication.
        """
        self.account_id = account_id
        self._platform = Platform.TIKTOK
        self._client: httpx.AsyncClient | None = None

    @property
    def platform(self) -> Platform:
        return self._platform

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def _get_access_token(self) -> tuple[str, str]:
        """Get valid access token and refresh if needed.

        Returns:
            Tuple of (access_token, open_id).

        Raises:
            TikTokOAuthError: If token refresh fails.
        """
        with get_session_context() as session:
            account_state = get_account_state(session, self.account_id)

            # Get open_id from metadata
            metadata = account_state.metadata_ or {}
            open_id = metadata.get("open_id", "")

            # Check if token needs refresh (1 hour before expiry)
            if account_state.token_expires_at:
                if account_state.token_expires_at < datetime.now(timezone.utc) + timedelta(hours=1):
                    logger.debug("tiktok_token_refresh", account_id=str(self.account_id))
                    try:
                        # Need refresh token from metadata
                        refresh_token = metadata.get("refresh_token", "")
                        if not refresh_token:
                            raise TikTokOAuthError("No refresh token available")

                        token_data = refresh_tiktok_token(refresh_token)
                        new_expires = datetime.now(timezone.utc) + timedelta(
                            seconds=token_data.get("expires_in", 86400)
                        )
                        update_account_tokens(
                            session, self.account_id, token_data["access_token"], new_expires
                        )

                        # Update open_id if provided
                        if token_data.get("open_id"):
                            open_id = token_data["open_id"]

                        return token_data["access_token"], open_id
                    except TikTokOAuthError as e:
                        if "expired" in str(e).lower() or "invalid" in str(e).lower():
                            mark_account_revoked(session, self.account_id, str(e))
                        raise

            return account_state.access_token, open_id

    async def fetch_metrics(self, platform_video_id: str) -> MetricsSnapshot:
        """Fetch current metrics for a TikTok video.

        Args:
            platform_video_id: The TikTok video/publish ID.

        Returns:
            MetricsSnapshot with current stats.
        """
        access_token, open_id = await self._get_access_token()
        client = await self._get_client()

        # Query video info
        response = await client.post(
            f"{TIKTOK_API_URL}/video/query/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={
                "filters": {
                    "video_ids": [platform_video_id],
                },
                "fields": [
                    "id",
                    "create_time",
                    "title",
                    "view_count",
                    "like_count",
                    "comment_count",
                    "share_count",
                    "duration",
                ],
            },
        )

        if response.status_code != 200:
            logger.error(
                "tiktok_video_api_error",
                status=response.status_code,
                body=response.text[:500],
            )
            raise RuntimeError(f"TikTok Video API error: {response.status_code}")

        data = response.json()

        if data.get("error", {}).get("code") != "ok":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            raise RuntimeError(f"TikTok API error: {error_msg}")

        videos = data.get("data", {}).get("videos", [])

        if not videos:
            raise ValueError(f"Video not found: {platform_video_id}")

        video = videos[0]

        views = video.get("view_count", 0)
        likes = video.get("like_count", 0)
        comments = video.get("comment_count", 0)
        shares = video.get("share_count", 0)
        duration = video.get("duration", 0)

        engagement_rate = 0.0
        if views > 0:
            engagement_rate = (likes + comments + shares) / views

        return MetricsSnapshot(
            platform=self.platform,
            platform_video_id=platform_video_id,
            fetched_at=datetime.now(timezone.utc),
            views=views,
            likes=likes,
            comments_count=comments,
            shares=shares,
            engagement_rate=engagement_rate,
            raw_data={
                "duration": duration,
                "title": video.get("title"),
                "create_time": video.get("create_time"),
            },
        )

    async def fetch_windowed_metrics(
        self,
        platform_video_id: str,
        publish_time: datetime,
    ) -> list[WindowedMetrics]:
        """Fetch metrics for all standard windows since publish.

        Note: TikTok API doesn't provide historical windowed data.
        We fetch current metrics and use them as lifetime data.

        Args:
            platform_video_id: The TikTok video ID.
            publish_time: When the video was published.

        Returns:
            List of WindowedMetrics (currently just lifetime metrics).
        """
        now = datetime.now(timezone.utc)
        windows = []

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
                "tiktok_metrics_error",
                video_id=platform_video_id,
                error=str(e),
            )

        return windows

    async def fetch_historical_metrics(
        self,
        platform_video_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[MetricsSnapshot]:
        """Fetch historical metrics for a date range.

        Note: TikTok API doesn't provide historical daily metrics.
        We return the current snapshot only.

        Args:
            platform_video_id: The TikTok video ID.
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
                "tiktok_historical_metrics_error",
                video_id=platform_video_id,
                error=str(e),
            )
            return []

    async def health_check(self) -> bool:
        """Check if TikTok API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            access_token, open_id = await self._get_access_token()

            client = await self._get_client()
            response = await client.get(
                f"{TIKTOK_API_URL}/user/info/",
                params={"fields": "open_id,display_name"},
                headers={"Authorization": f"Bearer {access_token}"},
            )

            return response.status_code == 200
        except Exception as e:
            logger.warning("tiktok_analytics_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
