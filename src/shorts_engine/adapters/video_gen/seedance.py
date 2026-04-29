"""Seedance 2.0 video generation provider via fal.ai.

Supports both text-to-video and image-to-video so storyboard-approved
boards can drive the real motion generation path.
"""

from __future__ import annotations

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


class SeedanceProvider(VideoGenProvider):
    """Seedance 2.0 video generation via fal.ai."""

    FAL_MODEL = "bytedance/seedance-2.0/text-to-video"
    FAL_MODEL_IMG2VID = "bytedance/seedance-2.0/image-to-video"
    FAL_MODEL_REF2VID = "bytedance/seedance-2.0/reference-to-video"
    DEFAULT_RESOLUTION = "720p"
    END_REFERENCE_IMAGE_OPTION = "end_reference_image"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or getattr(settings, "fal_api_key", None)
        if self.api_key:
            os.environ["FAL_KEY"] = self.api_key

        if not self.api_key and not os.environ.get("FAL_KEY"):
            logger.warning("FAL_KEY not configured for Seedance provider")

    @property
    def name(self) -> str:
        return "seedance"

    @property
    def supports_reference_images(self) -> bool:
        return True

    @staticmethod
    def _map_duration(duration_seconds: int) -> str:
        """Clamp to Seedance's supported 4-15 second range."""
        return str(max(4, min(15, duration_seconds)))

    @staticmethod
    def _map_aspect_ratio(aspect_ratio: str) -> str:
        ratio_map = {
            "vertical": "9:16",
            "horizontal": "16:9",
            "square": "1:1",
        }
        resolved = ratio_map.get(aspect_ratio, aspect_ratio)
        supported = {"auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
        return resolved if resolved in supported else "9:16"

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video using Seedance 2.0 via fal.ai."""
        if not self.api_key and not os.environ.get("FAL_KEY"):
            return VideoGenResult(success=False, error_message="FAL_KEY not configured")

        import fal_client

        full_prompt = request.prompt
        if request.style:
            full_prompt = f"{request.style}, {full_prompt}"
        if request.negative_prompt:
            full_prompt = f"{full_prompt}. Avoid: {request.negative_prompt}"

        duration = self._map_duration(request.duration_seconds)
        aspect_ratio = self._map_aspect_ratio(request.aspect_ratio or "9:16")
        options = dict(request.options or {})
        resolution = str(options.get("resolution") or self.DEFAULT_RESOLUTION)
        generate_audio = bool(options.get("generate_audio", False))
        seed = self._resolve_seed_option(options.get("seed"))
        reference_images = list(request.reference_images or [])

        has_reference = bool(reference_images)
        arguments: dict[str, Any] = {
            "prompt": full_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }
        if seed is not None:
            arguments["seed"] = seed

        if len(reference_images) > 1:
            model = self.FAL_MODEL_REF2VID
            arguments["image_urls"] = await self._upload_reference_images(reference_images)
            arguments["prompt"] = self._build_reference_prompt(full_prompt, len(reference_images))
        elif len(reference_images) == 1:
            model = self.FAL_MODEL_IMG2VID
            arguments["image_url"] = await self._upload_reference_image(reference_images[0])
            end_image_url = await self._resolve_end_reference_image_url(
                options.get(self.END_REFERENCE_IMAGE_OPTION)
            )
            if end_image_url:
                arguments["end_image_url"] = end_image_url
        else:
            model = self.FAL_MODEL

        logger.info(
            "seedance_generation_started",
            model=model,
            prompt_length=len(full_prompt),
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            has_reference=has_reference,
            reference_image_count=len(reference_images),
            has_end_reference_image=bool(arguments.get("end_image_url")),
            has_seed=seed is not None,
        )

        try:
            result = await fal_client.subscribe_async(model, arguments=arguments)
        except Exception as exc:
            logger.error("seedance_generation_error", error=str(exc))
            return VideoGenResult(success=False, error_message=str(exc))

        video_url = result.get("video", {}).get("url")
        if not video_url:
            logger.error("seedance_no_video_url", result_keys=list(result.keys()))
            return VideoGenResult(
                success=False,
                error_message="Seedance generation completed but no video URL returned",
            )

        duration_seconds = float(duration)
        metadata = {
            "provider": self.name,
            "video_url": video_url,
            "model": model,
            "seed": result.get("seed"),
            "request_id": result.get("requestId") or result.get("request_id"),
        }

        logger.info(
            "seedance_generation_completed",
            model=model,
            video_url=video_url[:100],
            duration_seconds=duration_seconds,
        )

        return VideoGenResult(
            success=True,
            video_data=None,
            duration_seconds=duration_seconds,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    async def _upload_reference_image(self, image_bytes: bytes) -> str:
        """Upload reference image bytes to fal storage and return the URL."""
        import fal_client

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file_handle:
            file_handle.write(image_bytes)
            temp_path = file_handle.name

        try:
            url = await fal_client.upload_file_async(temp_path)
            logger.info(
                "seedance_reference_image_uploaded",
                image_size=len(image_bytes),
                url=url[:100],
            )
            return url
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _upload_reference_images(self, reference_images: list[bytes]) -> list[str]:
        """Upload multiple reference images for Seedance reference-to-video."""
        return [await self._upload_reference_image(image_bytes) for image_bytes in reference_images]

    @staticmethod
    def _build_reference_prompt(prompt: str, image_count: int) -> str:
        """Guide Seedance to use uploaded reference images when multiple are available."""
        if any(f"@Image{index}" in prompt for index in range(1, image_count + 1)):
            return prompt

        if image_count == 2:
            prefix = (
                "@Image1 is the primary approved board / first-frame direction. "
                "@Image2 is an additional visual continuity reference."
            )
        else:
            prefix = (
                "@Image1 is the primary approved board / first-frame direction. "
                f"Use @Image2 through @Image{image_count} as additional visual continuity "
                "references."
            )
        return f"{prefix} {prompt}"

    @staticmethod
    def _resolve_seed_option(value: Any) -> int | None:
        """Normalize the optional Seedance seed value."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Seedance seed option must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            return int(value.strip())
        raise ValueError("Seedance seed option must be an integer")

    async def _resolve_end_reference_image_url(self, value: Any) -> str | None:
        """Resolve a minimal end-frame option into Seedance's end_image_url field."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (bytes, bytearray)):
            return await self._upload_reference_image(bytes(value))
        raise TypeError("Seedance end_reference_image option must be bytes or a URL string")

    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Check generation status through fal.ai."""
        try:
            import fal_client

            status = await fal_client.status_async(self.FAL_MODEL, job_id)
            return {"request_id": job_id, "status": str(status)}
        except Exception as exc:
            return {"error": str(exc)}

    async def health_check(self) -> bool:
        """Check if FAL_KEY is configured."""
        return bool(self.api_key or os.environ.get("FAL_KEY"))
