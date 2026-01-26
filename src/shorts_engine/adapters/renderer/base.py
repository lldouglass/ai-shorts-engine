"""Base interface for video rendering providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RenderRequest:
    """Request for video rendering."""

    video_data: bytes
    output_format: str = "mp4"
    resolution: str = "1080x1920"  # Vertical for Shorts
    fps: int = 30
    bitrate: str = "8M"
    audio_track: bytes | None = None
    watermark: bytes | None = None
    options: dict[str, Any] | None = None


@dataclass
class RenderResult:
    """Result from video rendering."""

    success: bool
    output_path: Path | None = None
    file_size_bytes: int | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class RendererProvider(ABC):
    """Abstract base class for video rendering providers.

    Implementations:
    - StubRendererProvider: Returns mock data for testing
    - FFmpegRenderer: Uses local FFmpeg for rendering (future)
    - CloudRenderer: Uses cloud transcoding service (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def render(self, request: RenderRequest) -> RenderResult:
        """Render video with the specified parameters.

        Args:
            request: Render request with video data and settings

        Returns:
            RenderResult with output path or error information
        """
        ...

    @abstractmethod
    async def get_video_info(self, video_path: Path) -> dict[str, Any]:
        """Get information about a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with video metadata (duration, resolution, etc.)
        """
        ...

    async def health_check(self) -> bool:
        """Check if the renderer is available and healthy.

        Returns:
            True if renderer is operational, False otherwise
        """
        return True
