"""Luma AI video generation provider."""

import asyncio
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


class LumaProvider(VideoGenProvider):
    """Luma AI (Dream Machine) video generation provider.

    Luma's API generates high-quality AI videos from text prompts.
    This provider handles async generation with polling for completion.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.lumalabs.ai/dream-machine/v1",
        poll_interval: float = 5.0,
        max_poll_attempts: int = 120,  # 10 minutes max
    ) -> None:
        self.api_key = api_key or getattr(settings, "luma_api_key", None)
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

        if not self.api_key:
            logger.warning("Luma API key not configured")

    @property
    def name(self) -> str:
        return "luma"

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video using Luma AI.

        This is an async operation:
        1. Submit generation request
        2. Poll for completion
        3. Return video URL when ready
        """
        if not self.api_key:
            return VideoGenResult(
                success=False,
                error_message="Luma API key not configured",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build the generation prompt
        full_prompt = request.prompt
        if request.style:
            full_prompt = f"{request.style}, {full_prompt}"

        payload = {
            "prompt": full_prompt,
            "aspect_ratio": request.aspect_ratio,
            "loop": False,
        }

        # Add options if provided
        if request.options:
            payload.update(request.options)

        logger.info(
            "luma_generation_started",
            prompt_length=len(full_prompt),
            aspect_ratio=request.aspect_ratio,
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Submit generation request
                response = await client.post(
                    f"{self.base_url}/generations",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            generation_id = data.get("id")
            if not generation_id:
                return VideoGenResult(
                    success=False,
                    error_message="No generation ID returned from Luma",
                )

            logger.info("luma_generation_submitted", generation_id=generation_id)

            # Poll for completion
            result = await self._poll_for_completion(generation_id, headers)
            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Luma API error: {e.response.status_code} - {e.response.text}"
            logger.error("luma_api_error", error=error_msg)
            return VideoGenResult(success=False, error_message=error_msg)
        except Exception as e:
            logger.error("luma_generation_error", error=str(e))
            return VideoGenResult(success=False, error_message=str(e))

    async def _poll_for_completion(
        self,
        generation_id: str,
        headers: dict[str, str],
    ) -> VideoGenResult:
        """Poll Luma API until generation completes."""
        for attempt in range(self.max_poll_attempts):
            await asyncio.sleep(self.poll_interval)

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{self.base_url}/generations/{generation_id}",
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                state = data.get("state", "unknown")
                logger.debug(
                    "luma_poll_status",
                    generation_id=generation_id,
                    state=state,
                    attempt=attempt + 1,
                )

                if state == "completed":
                    video_url = None
                    assets = data.get("assets", {})
                    if assets:
                        video_url = assets.get("video")

                    if not video_url:
                        return VideoGenResult(
                            success=False,
                            error_message="Generation completed but no video URL found",
                        )

                    logger.info(
                        "luma_generation_completed",
                        generation_id=generation_id,
                        video_url=video_url[:100],
                    )

                    return VideoGenResult(
                        success=True,
                        video_data=None,  # URL-based, not raw bytes
                        duration_seconds=5.0,  # Luma default
                        metadata={
                            "provider": self.name,
                            "generation_id": generation_id,
                            "video_url": video_url,
                            "state": state,
                            "raw_response": data,
                        },
                    )

                elif state == "failed":
                    failure_reason = data.get("failure_reason", "Unknown failure")
                    logger.error(
                        "luma_generation_failed",
                        generation_id=generation_id,
                        reason=failure_reason,
                    )
                    return VideoGenResult(
                        success=False,
                        error_message=f"Generation failed: {failure_reason}",
                    )

                # Still processing, continue polling

            except Exception as e:
                logger.warning(
                    "luma_poll_error",
                    generation_id=generation_id,
                    error=str(e),
                    attempt=attempt + 1,
                )
                # Continue polling on transient errors

        # Timeout
        return VideoGenResult(
            success=False,
            error_message=f"Generation timed out after {self.max_poll_attempts * self.poll_interval} seconds",
            metadata={"generation_id": generation_id},
        )

    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a generation job."""
        if not self.api_key:
            return {"error": "API key not configured"}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/generations/{job_id}",
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if Luma API is accessible."""
        if not self.api_key:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check generations list endpoint
                response = await client.get(
                    f"{self.base_url}/generations",
                    headers=headers,
                    params={"limit": 1},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error("luma_health_check_failed", error=str(e))
            return False
