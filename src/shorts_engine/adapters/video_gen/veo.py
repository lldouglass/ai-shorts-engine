"""Google Veo video generation provider."""

import asyncio
from typing import Any

from shorts_engine.adapters.video_gen.base import (
    VideoGenProvider,
    VideoGenRequest,
    VideoGenResult,
)
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class VeoProvider(VideoGenProvider):
    """Google Veo video generation via Gemini API.

    Uses the google-genai SDK to generate videos through Google's Veo model.
    Supports both veo-2.0-generate-001 (stable) and veo-3.1-fast-generate-preview (faster).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "veo-2.0-generate-001",
        poll_interval: float = 10.0,
        max_poll_attempts: int = 60,
    ) -> None:
        """Initialize the Veo provider.

        Args:
            api_key: Google API key for Gemini/Veo. Falls back to settings.
            model: Veo model to use. Defaults to veo-2.0-generate-001.
            poll_interval: Seconds between status polls.
            max_poll_attempts: Maximum polling attempts before timeout.
        """
        self.api_key = api_key or getattr(settings, "google_api_key", None)
        self.model = model
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self._client: Any = None

        if not self.api_key:
            logger.warning("Google API key not configured for Veo")

    def _get_client(self) -> Any:
        """Get or create the Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "veo"

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate a video using Google Veo.

        This is an async operation:
        1. Submit generation request
        2. Poll for completion
        3. Return video URL when ready

        Args:
            request: Video generation request with prompt and parameters

        Returns:
            VideoGenResult with video URL or error information
        """
        if not self.api_key:
            return VideoGenResult(
                success=False,
                error_message="Google API key not configured",
            )

        # Build the full prompt with style if provided
        full_prompt = request.prompt
        if request.style:
            full_prompt = f"{request.style}, {full_prompt}"

        # Map aspect ratio to Veo format (9:16 for shorts)
        aspect_ratio = request.aspect_ratio or "9:16"

        # Veo supports 4-8 seconds, use 8 for best quality
        duration_seconds = min(max(request.duration_seconds, 4), 8)

        logger.info(
            "veo_generation_started",
            prompt_length=len(full_prompt),
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            model=self.model,
        )

        try:
            # Run the synchronous SDK calls in a thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._generate_sync(full_prompt, aspect_ratio, duration_seconds),
            )
            return result

        except Exception as e:
            logger.error("veo_generation_error", error=str(e))
            return VideoGenResult(success=False, error_message=str(e))

    def _generate_sync(
        self,
        prompt: str,
        aspect_ratio: str,
        duration_seconds: int,
    ) -> VideoGenResult:
        """Synchronous video generation (runs in thread pool)."""
        from google.genai import types

        client = self._get_client()

        # Submit the generation request
        operation = client.models.generate_videos(
            model=self.model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                person_generation="allow_adult",
            ),
        )

        operation_name = operation.name
        logger.info("veo_generation_submitted", operation_name=operation_name)

        # Poll for completion
        for attempt in range(self.max_poll_attempts):
            import time

            time.sleep(self.poll_interval)

            operation = client.operations.get(operation=operation)

            logger.debug(
                "veo_poll_status",
                operation_name=operation_name,
                done=operation.done,
                attempt=attempt + 1,
            )

            if operation.done:
                if operation.error:
                    error_msg = f"Veo generation failed: {operation.error.message}"
                    logger.error("veo_generation_failed", error=error_msg)
                    return VideoGenResult(success=False, error_message=error_msg)

                # Extract video from response
                if operation.response and operation.response.generated_videos:
                    video = operation.response.generated_videos[0]
                    video_uri = video.video.uri

                    logger.info(
                        "veo_generation_completed",
                        operation_name=operation_name,
                        video_uri=video_uri[:100] if video_uri else None,
                    )

                    return VideoGenResult(
                        success=True,
                        video_data=None,  # URL-based, not raw bytes
                        duration_seconds=float(duration_seconds),
                        metadata={
                            "provider": self.name,
                            "operation_name": operation_name,
                            "video_url": video_uri,
                            "model": self.model,
                        },
                    )

                return VideoGenResult(
                    success=False,
                    error_message="Generation completed but no video returned",
                )

        # Timeout
        return VideoGenResult(
            success=False,
            error_message=(
                f"Generation timed out after {self.max_poll_attempts * self.poll_interval} seconds"
            ),
            metadata={"operation_name": operation_name},
        )

    async def check_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a generation job.

        Args:
            job_id: The operation name returned from generate()

        Returns:
            Status information including progress and state
        """
        if not self.api_key:
            return {"error": "API key not configured"}

        try:
            client = self._get_client()
            operation = client.operations.get(operation={"name": job_id})

            return {
                "name": operation.name,
                "done": operation.done,
                "error": str(operation.error) if operation.error else None,
                "has_response": operation.response is not None,
            }
        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if Veo API is accessible.

        Returns:
            True if provider is operational, False otherwise
        """
        if not self.api_key:
            return False

        try:
            # Try to initialize the client - this validates the API key format
            self._get_client()
            return True
        except Exception as e:
            logger.error("veo_health_check_failed", error=str(e))
            return False
