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
    Supports veo-2.0-generate-001 (stable), veo-3.1-generate-preview, and
    veo-3.1-fast-generate-preview (faster).

    Veo 3.1 features:
    - Reference images (up to 3) for character/style consistency
    - Negative prompts to avoid unwanted elements
    - Duration options: 4, 6, or 8 seconds (vs 5-8 for Veo 2.0)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "veo-3.1-generate-preview",
        poll_interval: float = 10.0,
        max_poll_attempts: int = 60,
    ) -> None:
        """Initialize the Veo provider.

        Args:
            api_key: Google API key for Gemini/Veo. Falls back to settings.
            model: Veo model to use. Defaults to veo-3.1-generate-preview.
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

        # Handle duration based on model version
        if "3.1" in self.model:
            # Veo 3.1 only supports 4, 6, or 8 seconds
            valid_durations = [4, 6, 8]
            duration_seconds = min(valid_durations, key=lambda x: abs(x - request.duration_seconds))
        else:
            # Veo 2.0 supports 5-8 seconds
            duration_seconds = min(max(request.duration_seconds, 5), 8)

        logger.info(
            "veo_generation_started",
            prompt_length=len(full_prompt),
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            model=self.model,
            has_reference_images=bool(request.reference_images),
            has_negative_prompt=bool(request.negative_prompt),
        )

        try:
            # Run the synchronous SDK calls in a thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._generate_sync(
                    full_prompt,
                    aspect_ratio,
                    duration_seconds,
                    request.negative_prompt,
                    request.reference_images,
                ),
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
        negative_prompt: str | None = None,
        reference_images: list[bytes] | None = None,
    ) -> VideoGenResult:
        """Synchronous video generation (runs in thread pool)."""
        from google.genai import types

        client = self._get_client()

        # Build reference images config if provided
        reference_images_config = None
        if reference_images and "3.1" in self.model:
            try:
                reference_images_config = [
                    types.VideoGenerationReferenceImage(
                        image=types.Image(
                            image_bytes=img,
                            mime_type="image/jpeg",
                        ),
                        reference_type="asset",
                    )
                    for img in reference_images[:3]  # Max 3 reference images
                ]
                logger.info(
                    "veo_reference_images_added",
                    count=len(reference_images_config),
                )
            except Exception as e:
                logger.warning(
                    "veo_reference_images_failed",
                    error=str(e),
                )
                reference_images_config = None

        # Build config with optional fields
        config_kwargs: dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "number_of_videos": 1,
            "duration_seconds": duration_seconds,
            "person_generation": "allow_adult",
        }

        # Add optional fields if provided
        if negative_prompt:
            config_kwargs["negative_prompt"] = negative_prompt
        if reference_images_config:
            config_kwargs["reference_images"] = reference_images_config

        # Submit the generation request
        operation = client.models.generate_videos(
            model=self.model,
            prompt=prompt,
            config=types.GenerateVideosConfig(**config_kwargs),
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
                            "download_headers": {
                                "x-goog-api-key": self.api_key,
                            },
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
