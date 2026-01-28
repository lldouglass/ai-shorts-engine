"""OpenAI DALL-E 3 image generation provider."""

import httpx

from shorts_engine.adapters.image_gen.base import (
    ImageGenProvider,
    ImageGenRequest,
    ImageGenResult,
    MotionParams,
)
from shorts_engine.config import get_settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class OpenAIDalleProvider(ImageGenProvider):
    """DALL-E 3 image generation via OpenAI API.

    Generates high-quality images optimized for anime/illustration styles.
    Cost: ~$0.04 per image (standard) or ~$0.08 per image (HD)
    """

    # DALL-E 3 supported sizes
    SUPPORTED_SIZES = {
        "9:16": "1024x1792",  # Vertical (Shorts) - HD only
        "16:9": "1792x1024",  # Horizontal - HD only
        "1:1": "1024x1024",  # Square
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "dall-e-3",
        default_quality: str = "hd",
        default_style: str = "vivid",
    ) -> None:
        """Initialize the DALL-E provider.

        Args:
            api_key: OpenAI API key. If None, uses config setting.
            model: Model to use (dall-e-3 or dall-e-2)
            default_quality: Default quality (standard or hd)
            default_style: Default style (vivid or natural)
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.default_quality = default_quality
        self.default_style = default_style
        self.base_url = "https://api.openai.com/v1/images/generations"

        if not self.api_key:
            logger.warning("OpenAI API key not configured for DALL-E provider")

    @property
    def name(self) -> str:
        return "dalle3"

    def _build_prompt(self, request: ImageGenRequest) -> str:
        """Build the full prompt with style tokens.

        Args:
            request: Image generation request

        Returns:
            Full prompt string
        """
        prompt_parts = []

        # Add style tokens if provided
        if request.style:
            prompt_parts.append(request.style)

        # Add main prompt
        prompt_parts.append(request.prompt)

        # Add quality hints for anime style
        if "anime" in request.prompt.lower() or "anime" in (request.style or "").lower():
            prompt_parts.append("high quality anime illustration, detailed linework")

        return ", ".join(prompt_parts)

    def _get_size(self, request: ImageGenRequest) -> str:
        """Get the appropriate size for the request.

        Args:
            request: Image generation request

        Returns:
            Size string (e.g., "1024x1792")
        """
        if request.size:
            return request.size

        # Map aspect ratio to DALL-E supported size
        size = self.SUPPORTED_SIZES.get(request.aspect_ratio)
        if not size:
            logger.warning(
                "unsupported_aspect_ratio",
                aspect_ratio=request.aspect_ratio,
                using_default="1024x1792",
            )
            size = "1024x1792"

        return size

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate an image using DALL-E 3.

        Args:
            request: Image generation request

        Returns:
            ImageGenResult with image URL
        """
        if not self.api_key:
            return ImageGenResult(
                success=False,
                error_message="OpenAI API key not configured",
            )

        full_prompt = self._build_prompt(request)
        size = self._get_size(request)
        quality = request.quality or self.default_quality

        logger.info(
            "dalle_generation_started",
            prompt_length=len(full_prompt),
            size=size,
            quality=quality,
            model=self.model,
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "n": 1,
                        "size": size,
                        "quality": quality,
                        "style": self.default_style,
                    },
                )

                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    logger.error(
                        "dalle_generation_failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    return ImageGenResult(
                        success=False,
                        error_message=f"DALL-E API error: {error_msg}",
                    )

                data = response.json()

        except httpx.TimeoutException:
            logger.error("dalle_generation_timeout")
            return ImageGenResult(
                success=False,
                error_message="DALL-E API timeout",
            )
        except Exception as e:
            logger.error("dalle_generation_exception", error=str(e))
            return ImageGenResult(
                success=False,
                error_message=f"DALL-E API exception: {str(e)}",
            )

        # Extract result
        image_data = data.get("data", [{}])[0]
        image_url = image_data.get("url")
        revised_prompt = image_data.get("revised_prompt")

        if not image_url:
            return ImageGenResult(
                success=False,
                error_message="No image URL in response",
            )

        # Determine motion based on style
        style_name = request.style or "default"
        # Extract style name from tokens if present
        if "attack on titan" in style_name.lower():
            style_name = "ATTACK_ON_TITAN"
        elif "dystopian" in style_name.lower():
            style_name = "DARK_DYSTOPIAN_ANIME"

        motion = MotionParams.for_style(style_name)

        logger.info(
            "dalle_generation_completed",
            image_url_length=len(image_url),
            revised_prompt_length=len(revised_prompt) if revised_prompt else 0,
        )

        return ImageGenResult(
            success=True,
            image_url=image_url,
            metadata={
                "provider": self.name,
                "model": self.model,
                "quality": quality,
                "size": size,
                "revised_prompt": revised_prompt,
                "original_prompt": request.prompt[:200],
            },
            suggested_motion=motion,
        )

    async def health_check(self) -> bool:
        """Check if the OpenAI API is accessible.

        Returns:
            True if API is reachable
        """
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False
