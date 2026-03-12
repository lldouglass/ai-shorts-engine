"""Seedance 2.0 video generation provider via xskill.ai API.

Supports both text-to-video and image-to-video (first_last_frames mode)
for frame chaining across scenes.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from shorts_engine.adapters.video_gen.base import (
    VideoGenProvider,
    VideoGenRequest,
    VideoGenResult,
)
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

# xskill.ai API endpoints
XSKILL_BASE_URL = "https://api.xskill.ai"
XSKILL_CREATE_URL = f"{XSKILL_BASE_URL}/api/v3/tasks/create"
XSKILL_QUERY_URL = f"{XSKILL_BASE_URL}/api/v3/tasks/query"

# Model identifiers
SEEDANCE_MODEL = "st-ai/super-seed2"
SEEDANCE_SPEED_FAST = "seedance_2.0_fast"
SEEDANCE_SPEED_STANDARD = "seedance_2.0"

# Polling settings
DEFAULT_POLL_INTERVAL = 5  # seconds
DEFAULT_POLL_TIMEOUT = 300  # 5 minutes


class SeedanceProvider(VideoGenProvider):
    """Seedance 2.0 video generation via xskill.ai API.

    Uses async HTTP requests with create + poll pattern.

    Modes:
    - first_last_frames: For frame chaining (scenes 2+ with reference image)
    - omni_reference: For multi-modal mixing (images, videos, audio)

    Duration: 4-15 seconds, passed as integer.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or getattr(settings, "seedance_api_key", None)
        if not self.api_key:
            self.api_key = os.environ.get("SEEDANCE_API_KEY")

        if not self.api_key:
            logger.warning("SEEDANCE_API_KEY not configured for Seedance provider")

    @property
    def name(self) -> str:
        return "seedance"

    @staticmethod
    def _map_duration(duration_seconds: int) -> int:
        """Clamp duration to Seedance's 4-15 second range."""
        return max(4, min(15, duration_seconds))

    @staticmethod
    def _map_aspect_ratio(aspect_ratio: str) -> str:
        """Map aspect ratio string to Seedance format."""
        # Seedance supports: 16:9, 9:16, 1:1
        ratio_map = {
            "16:9": "16:9",
            "9:16": "9:16",
            "1:1": "1:1",
            "vertical": "9:16",
            "horizontal": "16:9",
            "square": "1:1",
        }
        return ratio_map.get(aspect_ratio, "9:16")

    def _get_headers(self) -> dict[str, str]:
        """Build API request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video using Seedance 2.0 via xskill.ai.

        Uses create + poll pattern. For frame chaining (reference images),
        uses first_last_frames mode.

        Args:
            request: Video generation request with prompt and parameters.

        Returns:
            VideoGenResult with video_url in metadata.
        """
        if not self.api_key:
            return VideoGenResult(
                success=False,
                error_message="SEEDANCE_API_KEY not configured",
            )

        full_prompt = request.prompt
        if request.style:
            full_prompt = f"{request.style}, {full_prompt}"

        if request.negative_prompt:
            full_prompt = f"{full_prompt}. Avoid: {request.negative_prompt}"

        duration = self._map_duration(request.duration_seconds)
        aspect_ratio = self._map_aspect_ratio(request.aspect_ratio or "9:16")

        has_reference = bool(
            request.reference_images and len(request.reference_images) > 0
        )

        # Build params
        params: dict[str, Any] = {
            "model": SEEDANCE_SPEED_FAST,
            "prompt": full_prompt,
            "ratio": aspect_ratio,
            "duration": duration,
        }

        if has_reference:
            # Use first_last_frames mode for frame chaining
            # Upload the reference image first, then pass as filePaths
            image_url = await self._upload_reference_image(
                request.reference_images[0]
            )
            params["functionMode"] = "first_last_frames"
            params["filePaths"] = [image_url]
        else:
            params["functionMode"] = "first_last_frames"

        payload = {
            "model": SEEDANCE_MODEL,
            "params": params,
        }

        logger.info(
            "seedance_generation_started",
            prompt_length=len(full_prompt),
            duration=duration,
            aspect_ratio=aspect_ratio,
            has_reference=has_reference,
            mode=params["functionMode"],
        )

        try:
            # Step 1: Create task
            task_id = await self._create_task(payload)
            if not task_id:
                return VideoGenResult(
                    success=False,
                    error_message="Failed to create Seedance task",
                )

            logger.info(
                "seedance_task_created",
                task_id=task_id,
            )

            # Step 2: Poll for result
            video_url = await self._poll_for_result(task_id)
            if not video_url:
                return VideoGenResult(
                    success=False,
                    error_message="Seedance task completed but no video URL returned",
                )

            logger.info(
                "seedance_generation_completed",
                task_id=task_id,
                video_url=video_url[:100],
                duration_seconds=float(duration),
            )

            return VideoGenResult(
                success=True,
                video_data=None,
                duration_seconds=float(duration),
                metadata={
                    "provider": self.name,
                    "video_url": video_url,
                    "task_id": task_id,
                    "model": SEEDANCE_MODEL,
                },
            )

        except TimeoutError:
            logger.error("seedance_generation_timeout")
            return VideoGenResult(
                success=False,
                error_message="Seedance generation timed out",
            )
        except Exception as e:
            logger.error("seedance_generation_error", error=str(e))
            return VideoGenResult(success=False, error_message=str(e))

    async def _create_task(self, payload: dict[str, Any]) -> str | None:
        """Create a video generation task.

        Args:
            payload: Full request payload with model and params.

        Returns:
            Task ID string, or None on failure.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                XSKILL_CREATE_URL,
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                logger.error(
                    "seedance_create_failed",
                    status_code=response.status_code,
                    response_text=response.text[:500],
                )
                return None

            data = response.json()

            if data.get("code") != 200:
                logger.error(
                    "seedance_create_error",
                    code=data.get("code"),
                    message=data.get("message", "Unknown error"),
                )
                return None

            return data.get("data", {}).get("task_id")

    async def _poll_for_result(
        self,
        task_id: str,
        interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_POLL_TIMEOUT,
    ) -> str | None:
        """Poll for task completion and return video URL.

        Args:
            task_id: The task ID to poll.
            interval: Seconds between polls.
            timeout: Maximum seconds to wait.

        Returns:
            Video URL string, or None on failure.

        Raises:
            TimeoutError: If polling exceeds timeout.
            Exception: If task fails.
        """
        elapsed = 0
        async with httpx.AsyncClient(timeout=30) as client:
            while elapsed < timeout:
                response = await client.post(
                    XSKILL_QUERY_URL,
                    json={"task_id": task_id},
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    logger.warning(
                        "seedance_poll_http_error",
                        task_id=task_id,
                        status_code=response.status_code,
                    )
                    await asyncio.sleep(interval)
                    elapsed += interval
                    continue

                data = response.json()
                task_data = data.get("data", {})
                status = task_data.get("status", "unknown")

                logger.debug(
                    "seedance_poll_status",
                    task_id=task_id,
                    status=status,
                    elapsed=elapsed,
                )

                if status == "completed":
                    # Extract video URL from result
                    result = task_data.get("result", {})
                    output = result.get("output", {})
                    images = output.get("images", [])

                    if images:
                        return images[0]

                    # Try alternate response formats
                    video_url = output.get("video_url") or result.get("video_url")
                    if video_url:
                        return video_url

                    logger.error(
                        "seedance_no_video_url",
                        task_id=task_id,
                        result_keys=list(result.keys()),
                        output_keys=list(output.keys()),
                    )
                    return None

                elif status == "failed":
                    error_msg = task_data.get("error", "Task failed")
                    logger.error(
                        "seedance_task_failed",
                        task_id=task_id,
                        error=error_msg,
                    )
                    raise Exception(f"Seedance task failed: {error_msg}")

                # Still processing, wait and retry
                await asyncio.sleep(interval)
                elapsed += interval

        raise TimeoutError(
            f"Seedance task {task_id} did not complete within {timeout}s"
        )

    async def _upload_reference_image(self, image_bytes: bytes) -> str:
        """Upload reference image and return a URL.

        For xskill.ai, we need to provide a publicly accessible URL.
        We upload to a temporary hosting service or use base64 data URI.

        Args:
            image_bytes: JPEG/PNG image bytes.

        Returns:
            URL of the uploaded image.
        """
        # Save to temp file and try to use a data URI or upload endpoint
        # xskill.ai may accept base64 data URIs in filePaths
        import base64

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type
        if image_bytes[:4] == b"\x89PNG":
            mime = "image/png"
        else:
            mime = "image/jpeg"

        data_uri = f"data:{mime};base64,{b64}"

        logger.info(
            "seedance_reference_image_prepared",
            image_size=len(image_bytes),
            method="data_uri",
        )

        return data_uri

    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a Seedance generation task.

        Args:
            job_id: The task ID from xskill.ai.

        Returns:
            Status information dict.
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    XSKILL_QUERY_URL,
                    json={"task_id": job_id},
                    headers=self._get_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "task_id": job_id,
                        "status": data.get("data", {}).get("status", "unknown"),
                    }

                return {
                    "task_id": job_id,
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {"task_id": job_id, "error": str(e)}

    async def health_check(self) -> bool:
        """Check if the Seedance API key is configured."""
        return bool(self.api_key)
