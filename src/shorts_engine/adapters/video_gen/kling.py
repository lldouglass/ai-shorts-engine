"""Kling video generation provider via fal.ai.

Supports both text-to-video (Kling 2.6 Pro) and image-to-video (Kling O1)
for frame chaining across scenes.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from shorts_engine.adapters.video_gen.base import (
    VideoGenProvider,
    VideoGenRequest,
    VideoGenResult,
)
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class KlingProvider(VideoGenProvider):
    """Kling video generation via fal.ai.

    Uses the fal-client SDK with subscribe_async() for automatic polling.

    Models:
    - FAL_MODEL: Kling 2.6 Pro text-to-video (scene 1 or no reference image)
    - FAL_MODEL_IMG2VID: Kling O1 image-to-video (scenes 2+ with frame chaining)

    Duration mapping:
    - duration_seconds <= 5 → "5" (fal string format)
    - duration_seconds > 5  → "10"
    """

    FAL_MODEL = "fal-ai/kling-video/v2.6/pro/text-to-video"
    FAL_MODEL_IMG2VID = "fal-ai/kling-video/o1/image-to-video"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or getattr(settings, "fal_api_key", None)
        if self.api_key:
            os.environ["FAL_KEY"] = self.api_key

        if not self.api_key and not os.environ.get("FAL_KEY"):
            logger.warning("FAL_KEY not configured for Kling provider")

    @property
    def name(self) -> str:
        return "kling"

    @staticmethod
    def _map_duration(duration_seconds: int) -> str:
        """Map integer duration to fal's string duration format."""
        return "5" if duration_seconds <= 5 else "10"

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video using Kling 2.6 Pro via fal.ai.

        Uses fal_client.subscribe_async() which handles polling automatically.

        Args:
            request: Video generation request with prompt and parameters.

        Returns:
            VideoGenResult with video_url in metadata.
        """
        if not self.api_key and not os.environ.get("FAL_KEY"):
            return VideoGenResult(
                success=False,
                error_message="FAL_KEY not configured",
            )

        import fal_client

        full_prompt = request.prompt
        if request.style:
            full_prompt = f"{request.style}, {full_prompt}"

        duration = self._map_duration(request.duration_seconds)
        aspect_ratio = request.aspect_ratio or "9:16"
        negative_prompt = request.negative_prompt or "blur, distort, low quality"

        has_reference = bool(request.reference_images and len(request.reference_images) > 0)

        if has_reference:
            image_url = await self._upload_reference_image(request.reference_images[0])
            model = self.FAL_MODEL_IMG2VID
            arguments: dict[str, Any] = {
                "prompt": full_prompt,
                "start_image_url": image_url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
                "generate_audio": False,
            }
        else:
            model = self.FAL_MODEL
            arguments = {
                "prompt": full_prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
                "generate_audio": False,
            }

        logger.info(
            "kling_generation_started",
            model=model,
            prompt_length=len(full_prompt),
            duration=duration,
            aspect_ratio=aspect_ratio,
            has_reference=has_reference,
        )

        try:
            result = await fal_client.subscribe_async(
                model,
                arguments=arguments,
            )

            video_url = result.get("video", {}).get("url")
            if not video_url:
                logger.error("kling_no_video_url", result_keys=list(result.keys()))
                return VideoGenResult(
                    success=False,
                    error_message="Kling generation completed but no video URL returned",
                )

            duration_seconds = float(duration)

            logger.info(
                "kling_generation_completed",
                model=model,
                video_url=video_url[:100],
                duration_seconds=duration_seconds,
            )

            return VideoGenResult(
                success=True,
                video_data=None,
                duration_seconds=duration_seconds,
                metadata={
                    "provider": self.name,
                    "video_url": video_url,
                    "model": model,
                },
            )

        except Exception as e:
            logger.error("kling_generation_error", error=str(e))
            return VideoGenResult(success=False, error_message=str(e))

    async def _upload_reference_image(self, image_bytes: bytes) -> str:
        """Save image bytes to temp file and upload to fal CDN.

        Args:
            image_bytes: JPEG image bytes to upload.

        Returns:
            URL of the uploaded image on fal CDN.
        """
        import fal_client

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            url = await fal_client.upload_file_async(temp_path)
            logger.info(
                "kling_reference_image_uploaded",
                image_size=len(image_bytes),
                url=url[:100],
            )
            return url
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a fal.ai generation job.

        Args:
            job_id: The request ID returned from fal.ai.

        Returns:
            Status information dict.
        """
        try:
            import fal_client

            status = await fal_client.status_async(self.FAL_MODEL, job_id)
            return {"request_id": job_id, "status": str(status)}
        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if FAL_KEY is configured."""
        return bool(self.api_key or os.environ.get("FAL_KEY"))
