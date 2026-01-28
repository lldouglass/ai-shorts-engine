"""Base interface for image generation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MotionParams:
    """Motion parameters for Ken Burns / parallax effects.

    These parameters control how a static image is animated
    during video composition to create movement.
    """

    # Zoom (Ken Burns effect)
    zoom_start: float = 1.0  # 1.0 = 100% scale
    zoom_end: float = 1.1  # 1.1 = 110% scale (10% zoom in)

    # Pan (horizontal/vertical movement)
    pan_x_start: float = 0.0  # X offset start (-1.0 to 1.0)
    pan_x_end: float = 0.0  # X offset end
    pan_y_start: float = 0.0  # Y offset start
    pan_y_end: float = 0.0  # Y offset end

    # Timing
    ease: str = "ease-in-out"  # CSS easing function

    # Transition to next scene
    transition: str = "cut"  # cut, crossfade, wipe_left
    transition_duration: float = 0.3  # seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "zoom_start": self.zoom_start,
            "zoom_end": self.zoom_end,
            "pan_x_start": self.pan_x_start,
            "pan_x_end": self.pan_x_end,
            "pan_y_start": self.pan_y_start,
            "pan_y_end": self.pan_y_end,
            "ease": self.ease,
            "transition": self.transition,
            "transition_duration": self.transition_duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionParams":
        """Create from dictionary."""
        return cls(
            zoom_start=data.get("zoom_start", 1.0),
            zoom_end=data.get("zoom_end", 1.1),
            pan_x_start=data.get("pan_x_start", 0.0),
            pan_x_end=data.get("pan_x_end", 0.0),
            pan_y_start=data.get("pan_y_start", 0.0),
            pan_y_end=data.get("pan_y_end", 0.0),
            ease=data.get("ease", "ease-in-out"),
            transition=data.get("transition", "cut"),
            transition_duration=data.get("transition_duration", 0.3),
        )

    @classmethod
    def for_style(cls, style: str) -> "MotionParams":
        """Get recommended motion parameters for a visual style.

        Args:
            style: Style preset name

        Returns:
            MotionParams tuned for that style
        """
        style_upper = style.upper()

        if style_upper == "ATTACK_ON_TITAN":
            # Dramatic, intense motion
            return cls(
                zoom_end=1.15,
                ease="ease-out",
                transition="cut",
            )
        elif style_upper == "DARK_DYSTOPIAN_ANIME":
            # Slow, brooding movement
            return cls(
                zoom_end=1.08,
                pan_x_end=0.05,
                ease="ease-in-out",
            )
        elif style_upper == "CINEMATIC_REALISM":
            # Subtle, film-like motion
            return cls(
                zoom_end=1.05,
                ease="ease-in-out",
                transition="crossfade",
                transition_duration=0.5,
            )
        elif style_upper == "VIBRANT_MOTION_GRAPHICS":
            # Energetic, quick motion
            return cls(
                zoom_end=1.12,
                ease="ease-out",
                transition="cut",
            )

        # Default motion
        return cls()


@dataclass
class ImageGenRequest:
    """Request for image generation."""

    prompt: str
    style: str | None = None  # Style tokens to prepend
    aspect_ratio: str = "9:16"  # Vertical for Shorts
    quality: str = "hd"  # hd or standard
    size: str | None = None  # Override size (e.g., "1024x1792")
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageGenResult:
    """Result from image generation."""

    success: bool
    image_url: str | None = None
    image_data: bytes | None = None  # For providers that return raw bytes
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Suggested motion for this image type
    suggested_motion: MotionParams | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "image_url": self.image_url,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "suggested_motion": self.suggested_motion.to_dict() if self.suggested_motion else None,
        }


class ImageGenProvider(ABC):
    """Abstract base class for image generation providers.

    Implementations:
    - StubImageGenProvider: Returns placeholder images for testing
    - OpenAIDalleProvider: Uses DALL-E 3 for high-quality image generation
    - ReplicateFluxProvider: Uses Flux model via Replicate (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate an image from the given request.

        Args:
            request: Image generation request with prompt and parameters

        Returns:
            ImageGenResult with image URL or data
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if provider is operational
        """
        return True

    def get_aspect_ratio_size(self, aspect_ratio: str) -> str:
        """Convert aspect ratio to pixel dimensions.

        Args:
            aspect_ratio: Ratio string like "9:16"

        Returns:
            Size string like "1024x1792"
        """
        size_map = {
            "9:16": "1024x1792",  # Vertical (Shorts)
            "16:9": "1792x1024",  # Horizontal (Landscape)
            "1:1": "1024x1024",  # Square
            "4:3": "1024x768",  # Classic
            "3:4": "768x1024",  # Portrait
        }
        return size_map.get(aspect_ratio, "1024x1792")
