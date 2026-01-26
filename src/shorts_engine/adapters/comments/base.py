"""Base interface for comments ingestion adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shorts_engine.domain.enums import Platform


@dataclass
class CommentData:
    """Data for a single comment from a platform."""

    platform: Platform
    platform_comment_id: str
    platform_video_id: str
    author: str | None
    text: str
    likes: int = 0
    posted_at: datetime | None = None
    reply_to: str | None = None  # Parent comment ID if this is a reply
    raw_data: dict[str, Any] | None = None


class CommentsAdapter(ABC):
    """Abstract base class for comments ingestion adapters.

    Implementations:
    - StubCommentsAdapter: Returns mock data for testing
    - YouTubeCommentsAdapter: Fetches from YouTube Data API (future)
    - TikTokCommentsAdapter: Fetches from TikTok API (future)
    - InstagramCommentsAdapter: Fetches from Instagram API (future)
    """

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The platform this adapter fetches comments from."""
        ...

    @abstractmethod
    async def fetch_comments(
        self,
        platform_video_id: str,
        max_results: int = 100,
        since: datetime | None = None,
    ) -> list[CommentData]:
        """Fetch comments for a video.

        Args:
            platform_video_id: The video ID on the platform
            max_results: Maximum number of comments to fetch
            since: Only fetch comments after this timestamp

        Returns:
            List of CommentData objects
        """
        ...

    @abstractmethod
    async def reply_to_comment(
        self,
        platform_video_id: str,
        platform_comment_id: str,
        text: str,
    ) -> CommentData | None:
        """Reply to a comment (if supported by platform).

        Args:
            platform_video_id: The video ID on the platform
            platform_comment_id: The comment ID to reply to
            text: Reply text

        Returns:
            The created reply as CommentData, or None if failed
        """
        ...

    async def health_check(self) -> bool:
        """Check if the comments API is available.

        Returns:
            True if API is accessible, False otherwise
        """
        return True
