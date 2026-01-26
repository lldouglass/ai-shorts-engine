"""Stub renderer provider for testing."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from shorts_engine.adapters.renderer.base import RendererProvider, RenderRequest, RenderResult
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class StubRendererProvider(RendererProvider):
    """Stub provider that simulates video rendering without external dependencies."""

    @property
    def name(self) -> str:
        return "stub"

    async def render(self, request: RenderRequest) -> RenderResult:
        """Simulate video rendering with a delay."""
        logger.info(
            "stub_render_started",
            format=request.output_format,
            resolution=request.resolution,
        )

        # Simulate processing time
        await asyncio.sleep(0.3)

        # Create a temporary output file
        temp_dir = Path(tempfile.gettempdir()) / "shorts_engine"
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / f"rendered_{id(request)}.{request.output_format}"

        # Write stub content
        output_path.write_bytes(b"STUB_RENDERED_" + request.video_data[:50])

        file_size = output_path.stat().st_size

        logger.info(
            "stub_render_completed",
            output_path=str(output_path),
            file_size=file_size,
        )

        return RenderResult(
            success=True,
            output_path=output_path,
            file_size_bytes=file_size,
            duration_seconds=60.0,
            metadata={
                "provider": self.name,
                "format": request.output_format,
                "resolution": request.resolution,
            },
        )

    async def get_video_info(self, video_path: Path) -> dict[str, Any]:
        """Return stub video information."""
        return {
            "path": str(video_path),
            "duration_seconds": 60.0,
            "resolution": "1080x1920",
            "fps": 30,
            "codec": "h264",
            "bitrate": "8M",
        }

    async def health_check(self) -> bool:
        """Stub renderer is always healthy."""
        return True
