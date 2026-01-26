"""Base interface for video publishing adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shorts_engine.domain.enums import Platform


@dataclass
class PublishRequest:
    """Request to publish a video to a platform."""

    video_path: Path
    title: str
    description: str | None = None
    tags: list[str] | None = None
    thumbnail_path: Path | None = None
    scheduled_time: str | None = None  # ISO 8601 format
    visibility: str = "public"  # public, private, unlisted
    options: dict[str, Any] | None = None


@dataclass
class PublishResponse:
    """Response from publishing a video."""

    success: bool
    platform: Platform
    platform_video_id: str | None = None
    url: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class PublisherAdapter(ABC):
    """Abstract base class for platform publishing adapters.

    Implementations:
    - StubPublisherAdapter: Returns mock data for testing
    - YouTubeAdapter: Publishes to YouTube Shorts (future)
    - TikTokAdapter: Publishes to TikTok (future)
    - InstagramAdapter: Publishes to Instagram Reels (future)
    """

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The platform this adapter publishes to."""
        ...

    @abstractmethod
    async def publish(self, request: PublishRequest) -> PublishResponse:
        """Publish a video to the platform.

        Args:
            request: Publish request with video path and metadata

        Returns:
            PublishResponse with platform video ID and URL or error
        """
        ...

    @abstractmethod
    async def get_video_status(self, platform_video_id: str) -> dict[str, Any]:
        """Get the current status of a published video.

        Args:
            platform_video_id: The video ID on the platform

        Returns:
            Status information including processing state
        """
        ...

    @abstractmethod
    async def delete_video(self, platform_video_id: str) -> bool:
        """Delete a video from the platform.

        Args:
            platform_video_id: The video ID on the platform

        Returns:
            True if deletion was successful
        """
        ...

    async def health_check(self) -> bool:
        """Check if the publisher is available and authenticated.

        Returns:
            True if publisher is operational, False otherwise
        """
        return True
