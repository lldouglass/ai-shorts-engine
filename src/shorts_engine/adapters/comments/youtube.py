"""YouTube Comments adapter using Data API."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import httpx

from shorts_engine.adapters.comments.base import CommentsAdapter, CommentData
from shorts_engine.adapters.publisher.youtube_oauth import refresh_access_token, OAuthError
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger
from shorts_engine.services.accounts import (
    get_account_state,
    update_account_tokens,
    mark_account_revoked,
)

logger = get_logger(__name__)

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


class YouTubeCommentsAdapter(CommentsAdapter):
    """Fetches comments from YouTube Data API.

    Uses the commentThreads.list endpoint to fetch top-level comments.
    Reply functionality is not implemented (would require youtube.force-ssl scope).
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
            if account_state.token_expires_at:
                if account_state.token_expires_at < datetime.now(timezone.utc) + timedelta(minutes=5):
                    logger.debug("youtube_token_refresh", account_id=str(self.account_id))
                    try:
                        token_data = refresh_access_token(account_state.refresh_token)
                        new_expires = datetime.now(timezone.utc) + timedelta(
                            seconds=token_data["expires_in"]
                        )
                        update_account_tokens(
                            session, self.account_id, token_data["access_token"], new_expires
                        )
                        return token_data["access_token"]
                    except OAuthError as e:
                        if "invalid_grant" in str(e):
                            mark_account_revoked(session, self.account_id, str(e))
                        raise

            return account_state.access_token

    async def fetch_comments(
        self,
        platform_video_id: str,
        max_results: int = 100,
        since: datetime | None = None,
    ) -> list[CommentData]:
        """Fetch top-level comments for a video.

        Args:
            platform_video_id: The YouTube video ID.
            max_results: Maximum number of comments to fetch.
            since: Only fetch comments after this timestamp (optional).

        Returns:
            List of CommentData objects.
        """
        access_token = await self._get_access_token()
        client = await self._get_client()

        comments = []
        page_token = None

        while len(comments) < max_results:
            params: dict[str, Any] = {
                "part": "snippet",
                "videoId": platform_video_id,
                "maxResults": min(100, max_results - len(comments)),
                "order": "relevance",
                "textFormat": "plainText",
            }

            if page_token:
                params["pageToken"] = page_token

            response = await client.get(
                YOUTUBE_COMMENTS_URL,
                params=params,
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if response.status_code == 403:
                error_data = response.json()
                error_reason = error_data.get("error", {}).get("errors", [{}])[0].get("reason", "")

                if error_reason == "commentsDisabled":
                    logger.info(
                        "youtube_comments_disabled",
                        video_id=platform_video_id,
                    )
                    return []

                if "quotaExceeded" in str(error_data):
                    logger.error("youtube_comments_quota_exceeded")
                    raise RuntimeError("YouTube Data API quota exceeded")

                logger.warning(
                    "youtube_comments_forbidden",
                    video_id=platform_video_id,
                    reason=error_reason,
                )
                return []

            if response.status_code == 404:
                logger.warning(
                    "youtube_video_not_found",
                    video_id=platform_video_id,
                )
                return []

            if response.status_code != 200:
                logger.error(
                    "youtube_comments_api_error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                raise RuntimeError(f"YouTube Comments API error: {response.status_code}")

            data = response.json()

            for item in data.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})

                published_at = None
                if snippet.get("publishedAt"):
                    try:
                        published_at = datetime.fromisoformat(
                            snippet["publishedAt"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                # Filter by since if provided
                if since and published_at and published_at < since:
                    continue

                author_channel_id = snippet.get("authorChannelId", {}).get("value")
                reply_count = item.get("snippet", {}).get("totalReplyCount", 0)

                comments.append(
                    CommentData(
                        platform=self.platform,
                        platform_comment_id=item["id"],
                        platform_video_id=platform_video_id,
                        author=snippet.get("authorDisplayName"),
                        text=snippet.get("textDisplay", ""),
                        likes=int(snippet.get("likeCount", 0)),
                        posted_at=published_at,
                        raw_data={
                            "author_channel_id": author_channel_id,
                            "reply_count": reply_count,
                            "author_profile_image": snippet.get("authorProfileImageUrl"),
                            "updated_at": snippet.get("updatedAt"),
                        },
                    )
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.debug(
            "youtube_comments_fetched",
            video_id=platform_video_id,
            count=len(comments),
        )

        return comments

    async def reply_to_comment(
        self,
        platform_video_id: str,
        platform_comment_id: str,
        text: str,
    ) -> CommentData | None:
        """Reply to a comment.

        Note: Not implemented. Would require youtube.force-ssl scope
        and the comments.insert endpoint.

        Args:
            platform_video_id: The YouTube video ID.
            platform_comment_id: The comment ID to reply to.
            text: Reply text.

        Returns:
            None (not implemented).
        """
        logger.warning(
            "youtube_reply_not_implemented",
            video_id=platform_video_id,
            comment_id=platform_comment_id,
        )
        return None

    async def health_check(self) -> bool:
        """Check if YouTube Comments API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            await self._get_access_token()
            return True
        except Exception as e:
            logger.warning("youtube_comments_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
