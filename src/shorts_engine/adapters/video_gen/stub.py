"""Stub video generation provider for testing."""

import asyncio
from typing import Any
from uuid import uuid4

from shorts_engine.adapters.video_gen.base import (
    VideoGenProvider,
    VideoGenRequest,
    VideoGenResult,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class StubVideoGenProvider(VideoGenProvider):
    """Stub provider that simulates video generation without external calls."""

    @property
    def name(self) -> str:
        return "stub"

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Simulate video generation with a delay."""
        logger.info(
            "stub_video_generation_started",
            prompt=request.prompt[:100],
            duration=request.duration_seconds,
        )

        # Simulate processing time
        await asyncio.sleep(0.5)

        # Generate fake video data (just a marker)
        fake_video_data = b"STUB_VIDEO_DATA_" + request.prompt.encode()[:100]

        logger.info("stub_video_generation_completed", size=len(fake_video_data))

        return VideoGenResult(
            success=True,
            video_data=fake_video_data,
            duration_seconds=float(request.duration_seconds),
            metadata={
                "provider": self.name,
                "prompt": request.prompt,
                "job_id": str(uuid4()),
            },
        )

    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Return completed status for stub provider."""
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
        }

    async def health_check(self) -> bool:
        """Stub provider is always healthy."""
        return True
