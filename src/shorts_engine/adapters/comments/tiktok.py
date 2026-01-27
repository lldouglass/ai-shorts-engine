"""TikTok Comments adapter - Stub implementation.

TikTok's API does not currently provide access to video comments.
This is a stub implementation that returns empty results.

When TikTok adds comment API access in the future, this can be
replaced with a full implementation.
"""

from datetime import datetime
from uuid import UUID

from shorts_engine.adapters.comments.base import CommentData, CommentsAdapter
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class TikTokCommentsAdapter(CommentsAdapter):
    """Stub comments adapter for TikTok.

    TikTok does not currently provide API access to video comments.
    This adapter returns empty results and logs appropriate warnings.
    """

    def __init__(self, account_id: UUID) -> None:
        """Initialize the adapter.

        Args:
            account_id: The platform account UUID (unused for stub).
        """
        self.account_id = account_id
        self._platform = Platform.TIKTOK

    @property
    def platform(self) -> Platform:
        return self._platform

    async def fetch_comments(
        self,
        platform_video_id: str,  # noqa: ARG002
        max_results: int = 100,  # noqa: ARG002
        since: datetime | None = None,  # noqa: ARG002
    ) -> list[CommentData]:
        """Fetch comments for a TikTok video.

        Note: TikTok API does not provide access to video comments.
        This method always returns an empty list.

        Args:
            platform_video_id: The TikTok video ID.
            max_results: Maximum number of comments to fetch (ignored).
            since: Only fetch comments after this timestamp (ignored).

        Returns:
            Empty list - TikTok comments API not available.
        """
        logger.info(
            "tiktok_comments_not_available",
            video_id=platform_video_id,
            message="TikTok API does not provide access to video comments",
        )
        return []

    async def reply_to_comment(
        self,
        platform_video_id: str,
        platform_comment_id: str,
        _text: str,
    ) -> CommentData | None:
        """Reply to a TikTok comment.

        Note: TikTok API does not support replying to comments.
        This method always returns None.

        Args:
            platform_video_id: The TikTok video ID.
            platform_comment_id: The comment ID to reply to.
            text: Reply text.

        Returns:
            None - TikTok comments API not available.
        """
        logger.warning(
            "tiktok_reply_not_available",
            video_id=platform_video_id,
            comment_id=platform_comment_id,
            message="TikTok API does not support replying to comments",
        )
        return None

    async def health_check(self) -> bool:
        """Check if TikTok Comments API is accessible.

        Returns:
            True - Stub always reports healthy since there's nothing to check.
        """
        return True

    async def close(self) -> None:
        """Close any resources.

        No-op for stub implementation.
        """
        pass
