"""Instagram Comments adapter using Instagram Graph API."""

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import httpx

from shorts_engine.adapters.comments.base import CommentData, CommentsAdapter
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


class InstagramCommentsAdapter(CommentsAdapter):
    """Fetches comments from Instagram Graph API.

    Uses the Media Comments endpoint to fetch comments on Reels and posts.
    Supports pagination and filtering by timestamp.
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
            if (
                account_state.token_expires_at
                and account_state.token_expires_at < datetime.now(UTC) + timedelta(days=7)
            ):
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

    async def fetch_comments(
        self,
        platform_video_id: str,
        max_results: int = 100,
        since: datetime | None = None,
    ) -> list[CommentData]:
        """Fetch comments for an Instagram media.

        Args:
            platform_video_id: The Instagram media ID.
            max_results: Maximum number of comments to fetch.
            since: Only fetch comments after this timestamp.

        Returns:
            List of CommentData objects.
        """
        access_token = await self._get_access_token()
        client = await self._get_client()

        comments: list[CommentData] = []
        after_cursor = None

        while len(comments) < max_results:
            params: dict[str, Any] = {
                "fields": "id,text,timestamp,like_count,username,replies{id,text,timestamp,like_count,username}",
                "access_token": access_token,
                "limit": min(50, max_results - len(comments)),
            }

            if after_cursor:
                params["after"] = after_cursor

            response = await client.get(
                f"{GRAPH_API_URL}/{platform_video_id}/comments",
                params=params,
            )

            if response.status_code == 400:
                error_data = response.json()
                error = error_data.get("error", {})

                # Some media types don't support comments
                if error.get("code") == 100:
                    logger.info(
                        "instagram_comments_not_available",
                        video_id=platform_video_id,
                    )
                    return []

                logger.error(
                    "instagram_comments_api_error",
                    status=response.status_code,
                    error=error.get("message"),
                )
                return []

            if response.status_code != 200:
                logger.error(
                    "instagram_comments_api_error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                raise RuntimeError(f"Instagram Comments API error: {response.status_code}")

            data = response.json()

            for item in data.get("data", []):
                # Parse timestamp
                posted_at = None
                if item.get("timestamp"):
                    with contextlib.suppress(ValueError):
                        posted_at = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))

                # Filter by since timestamp
                if since and posted_at and posted_at < since:
                    continue

                comments.append(
                    CommentData(
                        platform=self.platform,
                        platform_comment_id=item["id"],
                        platform_video_id=platform_video_id,
                        author=item.get("username"),
                        text=item.get("text", ""),
                        likes=item.get("like_count", 0),
                        posted_at=posted_at,
                        raw_data={
                            "has_replies": bool(item.get("replies", {}).get("data")),
                            "reply_count": len(item.get("replies", {}).get("data", [])),
                        },
                    )
                )

                # Process replies
                for reply in item.get("replies", {}).get("data", []):
                    reply_posted_at = None
                    if reply.get("timestamp"):
                        with contextlib.suppress(ValueError):
                            reply_posted_at = datetime.fromisoformat(
                                reply["timestamp"].replace("Z", "+00:00")
                            )

                    if since and reply_posted_at and reply_posted_at < since:
                        continue

                    comments.append(
                        CommentData(
                            platform=self.platform,
                            platform_comment_id=reply["id"],
                            platform_video_id=platform_video_id,
                            author=reply.get("username"),
                            text=reply.get("text", ""),
                            likes=reply.get("like_count", 0),
                            posted_at=reply_posted_at,
                            reply_to=item["id"],
                            raw_data={"is_reply": True},
                        )
                    )

            # Handle pagination
            paging = data.get("paging", {})
            cursors = paging.get("cursors", {})
            after_cursor = cursors.get("after")

            if not after_cursor or not paging.get("next"):
                break

        logger.debug(
            "instagram_comments_fetched",
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
        """Reply to a comment on Instagram.

        Args:
            platform_video_id: The Instagram media ID.
            platform_comment_id: The comment ID to reply to.
            text: Reply text.

        Returns:
            The created reply as CommentData, or None if failed.
        """
        access_token = await self._get_access_token()
        client = await self._get_client()

        response = await client.post(
            f"{GRAPH_API_URL}/{platform_comment_id}/replies",
            params={
                "message": text,
                "access_token": access_token,
            },
        )

        if response.status_code != 200:
            logger.error(
                "instagram_reply_failed",
                comment_id=platform_comment_id,
                status=response.status_code,
                body=response.text[:500],
            )
            return None

        data = response.json()
        reply_id = data.get("id")

        if not reply_id:
            return None

        return CommentData(
            platform=self.platform,
            platform_comment_id=reply_id,
            platform_video_id=platform_video_id,
            author=None,  # Our account
            text=text,
            likes=0,
            posted_at=datetime.now(UTC),
            reply_to=platform_comment_id,
            raw_data={"is_our_reply": True},
        )

    async def health_check(self) -> bool:
        """Check if Instagram Comments API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            await self._get_access_token()
            return True
        except Exception as e:
            logger.warning("instagram_comments_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
