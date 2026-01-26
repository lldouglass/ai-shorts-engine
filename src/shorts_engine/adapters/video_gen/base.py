"""Base interface for video generation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VideoGenResult:
    """Result from video generation."""

    success: bool
    video_data: bytes | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    duration_seconds: float | None = None


@dataclass
class VideoGenRequest:
    """Request for video generation."""

    prompt: str
    duration_seconds: int = 60
    style: str | None = None
    aspect_ratio: str = "9:16"  # Vertical for Shorts
    options: dict[str, Any] | None = None


class VideoGenProvider(ABC):
    """Abstract base class for video generation providers.

    Implementations:
    - StubVideoGenProvider: Returns mock data for testing
    - OpenAISoraProvider: Uses OpenAI's Sora API (future)
    - RunwayProvider: Uses Runway ML API (future)
    - PikaProvider: Uses Pika Labs API (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video from the given request.

        Args:
            request: Video generation request with prompt and parameters

        Returns:
            VideoGenResult with video data or error information
        """
        ...

    @abstractmethod
    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of an async generation job.

        Args:
            job_id: The job ID returned from generate()

        Returns:
            Status information including progress and state
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is available and healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        return True
