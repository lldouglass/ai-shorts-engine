"""Stub image generation provider for testing."""

import asyncio
import random
from uuid import uuid4

from shorts_engine.adapters.image_gen.base import (
    ImageGenProvider,
    ImageGenRequest,
    ImageGenResult,
    MotionParams,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

# Placeholder image URLs for testing (various anime-style placeholders)
PLACEHOLDER_IMAGES = [
    "https://placehold.co/1024x1792/1a1a1a/ffffff?text=Scene+1",
    "https://placehold.co/1024x1792/2d2d2d/ffffff?text=Scene+2",
    "https://placehold.co/1024x1792/4a4a4a/ffffff?text=Scene+3",
    "https://placehold.co/1024x1792/1a1a2e/ffffff?text=Scene+4",
    "https://placehold.co/1024x1792/2e1a1a/ffffff?text=Scene+5",
]


class StubImageGenProvider(ImageGenProvider):
    """Stub provider that returns placeholder images for testing.

    Simulates image generation without making API calls.
    Useful for testing pipelines and development.
    """

    def __init__(self, latency_ms: int = 100) -> None:
        """Initialize the stub provider.

        Args:
            latency_ms: Simulated latency in milliseconds
        """
        self.latency_ms = latency_ms

    @property
    def name(self) -> str:
        return "stub"

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Return a placeholder image for testing.

        Args:
            request: Image generation request (prompt used for logging only)

        Returns:
            ImageGenResult with placeholder image URL
        """
        # Simulate API latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Generate a unique placeholder URL
        image_id = str(uuid4())[:8]
        size = request.size or self.get_aspect_ratio_size(request.aspect_ratio)
        width, height = size.split("x")

        # Create a placeholder URL with prompt text
        prompt_short = request.prompt[:30].replace(" ", "+") if request.prompt else "image"
        placeholder_url = f"https://placehold.co/{width}x{height}/1a1a1a/ffffff?text={prompt_short}"

        # Alternatively, use one of the predefined placeholders
        if random.random() > 0.5:
            placeholder_url = random.choice(PLACEHOLDER_IMAGES)

        # Determine motion based on style
        style = request.style or "default"
        motion = MotionParams.for_style(style)

        logger.info(
            "stub_image_generated",
            prompt_length=len(request.prompt),
            style=request.style,
            aspect_ratio=request.aspect_ratio,
            image_id=image_id,
        )

        return ImageGenResult(
            success=True,
            image_url=placeholder_url,
            metadata={
                "provider": self.name,
                "prompt": request.prompt[:100],
                "style": request.style,
                "aspect_ratio": request.aspect_ratio,
                "image_id": image_id,
                "is_placeholder": True,
            },
            suggested_motion=motion,
        )

    async def health_check(self) -> bool:
        """Stub provider is always healthy."""
        return True
