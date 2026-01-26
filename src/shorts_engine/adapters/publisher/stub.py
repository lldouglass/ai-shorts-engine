"""Stub publisher adapter for testing."""

import asyncio
from typing import Any
from uuid import uuid4

from shorts_engine.adapters.publisher.base import (
    PublisherAdapter,
    PublishRequest,
    PublishResponse,
)
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class StubPublisherAdapter(PublisherAdapter):
    """Stub adapter that simulates publishing without external calls."""

    def __init__(self, platform: Platform = Platform.YOUTUBE) -> None:
        self._platform = platform

    @property
    def platform(self) -> Platform:
        return self._platform

    async def publish(self, request: PublishRequest) -> PublishResponse:
        """Simulate video publishing with a delay."""
        logger.info(
            "stub_publish_started",
            platform=self.platform,
            title=request.title,
        )

        # Simulate upload time
        await asyncio.sleep(0.5)

        platform_video_id = f"stub_{uuid4().hex[:12]}"
        url = f"https://{self.platform}.example.com/shorts/{platform_video_id}"

        logger.info(
            "stub_publish_completed",
            platform=self.platform,
            platform_video_id=platform_video_id,
            url=url,
        )

        return PublishResponse(
            success=True,
            platform=self.platform,
            platform_video_id=platform_video_id,
            url=url,
            metadata={
                "adapter": "stub",
                "title": request.title,
                "visibility": request.visibility,
            },
        )

    async def get_video_status(self, platform_video_id: str) -> dict[str, Any]:
        """Return completed status for stub adapter."""
        return {
            "platform_video_id": platform_video_id,
            "status": "published",
            "processing_status": "complete",
            "visibility": "public",
        }

    async def delete_video(self, platform_video_id: str) -> bool:
        """Simulate video deletion."""
        logger.info(
            "stub_delete_video",
            platform=self.platform,
            platform_video_id=platform_video_id,
        )
        await asyncio.sleep(0.1)
        return True

    async def health_check(self) -> bool:
        """Stub adapter is always healthy."""
        return True
