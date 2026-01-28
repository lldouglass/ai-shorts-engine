"""Creatomate video rendering provider with dynamic composition."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from shorts_engine.adapters.image_gen.base import MotionParams
from shorts_engine.adapters.renderer.base import RendererProvider, RenderRequest, RenderResult
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SceneClip:
    """A scene clip for the composition."""

    video_url: str
    duration_seconds: float
    caption_text: str | None = None
    scene_number: int = 0


@dataclass
class ImageSceneClip:
    """An image-based scene clip with Ken Burns motion.

    Used for "limited animation" style where static images
    are animated with zoom/pan effects.
    """

    image_url: str
    duration_seconds: float
    motion: MotionParams = field(default_factory=MotionParams)
    caption_text: str | None = None
    scene_number: int = 0
    transition: str = "cut"  # cut, crossfade, fade_to_black
    transition_duration: float = 0.3


@dataclass
class CreatomateRenderRequest:
    """Extended render request for Creatomate with scene data."""

    scenes: list[SceneClip]
    voiceover_url: str | None = None
    output_format: str = "mp4"
    width: int = 1080
    height: int = 1920
    fps: int = 30
    background_music_url: str | None = None
    background_music_volume: float = 0.3
    caption_style: dict[str, Any] | None = None


@dataclass
class ImageCompositionRequest:
    """Render request for image-based compositions with Ken Burns effects.

    Used for "limited animation" style videos where static images
    are animated with motion effects.
    """

    images: list[ImageSceneClip]
    voiceover_url: str | None = None
    output_format: str = "mp4"
    width: int = 1080
    height: int = 1920
    fps: int = 30
    background_music_url: str | None = None
    background_music_volume: float = 0.3
    caption_style: dict[str, Any] | None = None


class CreatomateProvider(RendererProvider):
    """Creatomate API provider for video composition and rendering.

    Uses dynamic composition payloads to stitch clips, add captions,
    and mix voiceover/music into a final video.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.creatomate.com/v1",
        poll_interval: float = 5.0,
        max_poll_attempts: int = 120,  # 10 minutes max
        webhook_url: str | None = None,
    ) -> None:
        self.api_key = api_key or getattr(settings, "creatomate_api_key", None)
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self.webhook_url = webhook_url or getattr(settings, "creatomate_webhook_url", None)

        if not self.api_key:
            logger.warning("Creatomate API key not configured")

    @property
    def name(self) -> str:
        return "creatomate"

    async def render(self, _request: RenderRequest) -> RenderResult:
        """Render using basic RenderRequest (legacy interface)."""
        # This is a simplified version - use render_composition for full features
        return RenderResult(
            success=False,
            error_message="Use render_composition() for Creatomate rendering",
        )

    async def render_composition(
        self,
        request: CreatomateRenderRequest,
    ) -> RenderResult:
        """Render a composition using Creatomate's dynamic API.

        Args:
            request: Extended render request with scene clips and options

        Returns:
            RenderResult with output URL or error information
        """
        if not self.api_key:
            return RenderResult(
                success=False,
                error_message="Creatomate API key not configured",
            )

        # Build the dynamic composition payload
        payload = self._build_composition_payload(request)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "creatomate_render_started",
            scene_count=len(request.scenes),
            has_voiceover=request.voiceover_url is not None,
            output_size=f"{request.width}x{request.height}",
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/renders",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            # Handle response - could be single render or array
            renders = data if isinstance(data, list) else [data]
            render_info = renders[0]
            render_id = render_info.get("id")

            if not render_id:
                return RenderResult(
                    success=False,
                    error_message="No render ID returned from Creatomate",
                )

            logger.info("creatomate_render_submitted", render_id=render_id)

            # Poll for completion or use webhook
            if self.webhook_url:
                # Return early with render_id for webhook-based completion
                return RenderResult(
                    success=True,
                    metadata={
                        "render_id": render_id,
                        "status": "processing",
                        "webhook_pending": True,
                    },
                )

            # Poll for completion
            result = await self._poll_for_completion(render_id, headers)
            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Creatomate API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"{error_msg} - {error_data}"
            except Exception:
                pass
            logger.error("creatomate_api_error", error=error_msg)
            return RenderResult(success=False, error_message=error_msg)
        except Exception as e:
            logger.error("creatomate_render_error", error=str(e))
            return RenderResult(success=False, error_message=str(e))

    async def render_image_composition(
        self,
        request: ImageCompositionRequest,
    ) -> RenderResult:
        """Render an image-based composition with Ken Burns effects.

        This renders static images with zoom/pan animations to create
        the "limited animation" style used in anime.

        Args:
            request: Image composition request with images and motion params

        Returns:
            RenderResult with output URL or error information
        """
        if not self.api_key:
            return RenderResult(
                success=False,
                error_message="Creatomate API key not configured",
            )

        # Build the image composition payload
        payload = self._build_image_composition_payload(request)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "creatomate_image_render_started",
            image_count=len(request.images),
            has_voiceover=request.voiceover_url is not None,
            output_size=f"{request.width}x{request.height}",
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/renders",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            # Handle response
            renders = data if isinstance(data, list) else [data]
            render_info = renders[0]
            render_id = render_info.get("id")

            if not render_id:
                return RenderResult(
                    success=False,
                    error_message="No render ID returned from Creatomate",
                )

            logger.info("creatomate_image_render_submitted", render_id=render_id)

            # Poll for completion or use webhook
            if self.webhook_url:
                return RenderResult(
                    success=True,
                    metadata={
                        "render_id": render_id,
                        "status": "processing",
                        "webhook_pending": True,
                        "composition_type": "image_sequence",
                    },
                )

            # Poll for completion
            result = await self._poll_for_completion(render_id, headers)
            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Creatomate API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"{error_msg} - {error_data}"
            except Exception:
                pass
            logger.error("creatomate_image_api_error", error=error_msg)
            return RenderResult(success=False, error_message=error_msg)
        except Exception as e:
            logger.error("creatomate_image_render_error", error=str(e))
            return RenderResult(success=False, error_message=str(e))

    def _build_image_composition_payload(
        self,
        request: ImageCompositionRequest,
    ) -> dict[str, Any]:
        """Build the image composition payload with Ken Burns effects.

        Creates a composition with:
        - Sequenced images with zoom/pan animations
        - Transitions between images
        - Burned-in captions
        - Optional voiceover and background music
        """
        total_duration = sum(img.duration_seconds for img in request.images)
        elements: list[dict[str, Any]] = []
        current_time = 0.0

        for img in request.images:
            motion = img.motion

            # Calculate animation keyframes for Ken Burns effect
            # Creatomate uses keyframes for animations
            image_element: dict[str, Any] = {
                "type": "image",
                "source": img.image_url,
                "time": current_time,
                "duration": img.duration_seconds,
                "fit": "cover",
                # Ken Burns animation via keyframes
                "animations": [
                    {
                        "easing": motion.ease,
                        "type": "scale",
                        "scope": "element",
                        "start_scale": f"{motion.zoom_start * 100}%",
                        "end_scale": f"{motion.zoom_end * 100}%",
                    },
                ],
            }

            # Add pan animation if specified
            if motion.pan_x_start != motion.pan_x_end or motion.pan_y_start != motion.pan_y_end:
                # Convert pan to x/y offset percentages
                x_offset_start = f"{50 + motion.pan_x_start * 100}%"
                x_offset_end = f"{50 + motion.pan_x_end * 100}%"
                y_offset_start = f"{50 + motion.pan_y_start * 100}%"
                y_offset_end = f"{50 + motion.pan_y_end * 100}%"

                image_element["x_anchor"] = x_offset_start
                image_element["y_anchor"] = y_offset_start
                image_element["animations"].append(
                    {
                        "easing": motion.ease,
                        "type": "pan",
                        "x_anchor": [x_offset_start, x_offset_end],
                        "y_anchor": [y_offset_start, y_offset_end],
                    }
                )

            # Add transition effect
            if img.transition == "crossfade" and current_time > 0:
                image_element["enter"] = {
                    "type": "fade",
                    "duration": img.transition_duration,
                }
            elif img.transition == "fade_to_black":
                image_element["exit"] = {
                    "type": "fade",
                    "duration": img.transition_duration,
                    "background_color": "#000000",
                }

            elements.append(image_element)

            # Caption element
            if img.caption_text:
                caption_style = request.caption_style or self._default_caption_style()
                caption_element: dict[str, Any] = {
                    "type": "text",
                    "text": img.caption_text,
                    "time": current_time,
                    "duration": img.duration_seconds,
                    **caption_style,
                }
                elements.append(caption_element)

            current_time += img.duration_seconds

        # Audio tracks
        if request.voiceover_url:
            elements.append(
                {
                    "type": "audio",
                    "source": request.voiceover_url,
                    "time": 0,
                    "duration": total_duration,
                    "volume": "100%",
                }
            )

        if request.background_music_url:
            elements.append(
                {
                    "type": "audio",
                    "source": request.background_music_url,
                    "time": 0,
                    "duration": total_duration,
                    "volume": f"{int(request.background_music_volume * 100)}%",
                    "audio_fade_out": 2.0,
                }
            )

        payload: dict[str, Any] = {
            "output_format": request.output_format,
            "width": request.width,
            "height": request.height,
            "frame_rate": request.fps,
            "duration": total_duration,
            "elements": elements,
        }

        if self.webhook_url:
            payload["webhook_url"] = self.webhook_url

        return payload

    def _build_composition_payload(
        self,
        request: CreatomateRenderRequest,
    ) -> dict[str, Any]:
        """Build the dynamic composition payload for Creatomate.

        This creates a composition with:
        - Sequenced video clips
        - Burned-in captions
        - Optional voiceover track
        - Optional background music
        """
        # Calculate total duration
        total_duration = sum(s.duration_seconds for s in request.scenes)

        # Build the elements list
        elements: list[dict[str, Any]] = []

        # Track for video clips (sequential composition)
        video_track: list[dict[str, Any]] = []
        current_time = 0.0

        for scene in request.scenes:
            # Video clip element
            clip_element: dict[str, Any] = {
                "type": "video",
                "source": scene.video_url,
                "time": current_time,
                "duration": scene.duration_seconds,
                "fit": "cover",
            }
            video_track.append(clip_element)

            # Caption element (burned-in text)
            if scene.caption_text:
                caption_style = request.caption_style or self._default_caption_style()
                caption_element: dict[str, Any] = {
                    "type": "text",
                    "text": scene.caption_text,
                    "time": current_time,
                    "duration": scene.duration_seconds,
                    **caption_style,
                }
                elements.append(caption_element)

            current_time += scene.duration_seconds

        # Add video track elements
        elements.extend(video_track)

        # Audio tracks
        audio_elements: list[dict[str, Any]] = []

        # Voiceover track
        if request.voiceover_url:
            audio_elements.append(
                {
                    "type": "audio",
                    "source": request.voiceover_url,
                    "time": 0,
                    "duration": total_duration,
                    "volume": "100%",
                }
            )

        # Background music (lower volume)
        if request.background_music_url:
            audio_elements.append(
                {
                    "type": "audio",
                    "source": request.background_music_url,
                    "time": 0,
                    "duration": total_duration,
                    "volume": f"{int(request.background_music_volume * 100)}%",
                    "audio_fade_out": 2.0,
                }
            )

        elements.extend(audio_elements)

        # Build the full payload
        payload: dict[str, Any] = {
            "output_format": request.output_format,
            "width": request.width,
            "height": request.height,
            "frame_rate": request.fps,
            "duration": total_duration,
            "elements": elements,
        }

        # Add webhook if configured
        if self.webhook_url:
            payload["webhook_url"] = self.webhook_url

        return payload

    def _default_caption_style(self) -> dict[str, Any]:
        """Get default caption styling for burned-in text."""
        return {
            "y": "85%",  # Position near bottom
            "width": "90%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "font_family": "Montserrat",
            "font_weight": "800",
            "font_size": "7 vmin",
            "fill_color": "#ffffff",
            "stroke_color": "#000000",
            "stroke_width": "1.5 vmin",
            "text_transform": "uppercase",
            "shadow_color": "rgba(0,0,0,0.8)",
            "shadow_blur": "4 vmin",
        }

    async def _poll_for_completion(
        self,
        render_id: str,
        headers: dict[str, str],
    ) -> RenderResult:
        """Poll Creatomate API until render completes."""
        for attempt in range(self.max_poll_attempts):
            await asyncio.sleep(self.poll_interval)

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{self.base_url}/renders/{render_id}",
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                status = data.get("status", "unknown")
                logger.debug(
                    "creatomate_poll_status",
                    render_id=render_id,
                    status=status,
                    attempt=attempt + 1,
                )

                if status == "succeeded":
                    output_url = data.get("url")
                    if not output_url:
                        return RenderResult(
                            success=False,
                            error_message="Render succeeded but no URL returned",
                        )

                    logger.info(
                        "creatomate_render_completed",
                        render_id=render_id,
                        output_url=output_url[:100],
                    )

                    return RenderResult(
                        success=True,
                        file_size_bytes=data.get("file_size"),
                        duration_seconds=data.get("duration"),
                        metadata={
                            "provider": self.name,
                            "render_id": render_id,
                            "url": output_url,
                            "status": status,
                        },
                    )

                elif status == "failed":
                    error_msg = data.get("error_message", "Unknown render failure")
                    logger.error(
                        "creatomate_render_failed",
                        render_id=render_id,
                        error=error_msg,
                    )
                    return RenderResult(
                        success=False,
                        error_message=f"Render failed: {error_msg}",
                    )

                # Still processing, continue polling

            except Exception as e:
                logger.warning(
                    "creatomate_poll_error",
                    render_id=render_id,
                    error=str(e),
                    attempt=attempt + 1,
                )
                # Continue polling on transient errors

        # Timeout
        return RenderResult(
            success=False,
            error_message=f"Render timed out after {self.max_poll_attempts * self.poll_interval} seconds",
            metadata={"render_id": render_id},
        )

    async def check_render_status(self, render_id: str) -> dict[str, Any]:
        """Check the status of a render job (for webhook scenarios)."""
        if not self.api_key:
            return {"error": "API key not configured"}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/renders/{render_id}",
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
        except Exception as e:
            return {"error": str(e)}

    async def get_video_info(self, video_path: Path) -> dict[str, Any]:
        """Get information about a video file (not used with cloud rendering)."""
        return {
            "path": str(video_path),
            "note": "Video info not available for cloud-rendered files",
        }

    async def health_check(self) -> bool:
        """Check if Creatomate API is accessible."""
        if not self.api_key:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check renders list endpoint
                response = await client.get(
                    f"{self.base_url}/renders",
                    headers=headers,
                    params={"limit": 1},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error("creatomate_health_check_failed", error=str(e))
            return False


def build_creatomate_payload(
    scenes: list[SceneClip],
    voiceover_url: str | None = None,
    background_music_url: str | None = None,
    caption_style: dict[str, Any] | None = None,
    width: int = 1080,
    height: int = 1920,
) -> dict[str, Any]:
    """Build a Creatomate composition payload (for testing/external use).

    This is a standalone function that can be unit tested independently.

    Args:
        scenes: List of scene clips to stitch
        voiceover_url: Optional voiceover audio URL
        background_music_url: Optional background music URL
        caption_style: Optional custom caption styling
        width: Output width (default 1080)
        height: Output height (default 1920)

    Returns:
        Dictionary payload for Creatomate API
    """
    request = CreatomateRenderRequest(
        scenes=scenes,
        voiceover_url=voiceover_url,
        background_music_url=background_music_url,
        caption_style=caption_style,
        width=width,
        height=height,
    )

    provider = CreatomateProvider(api_key="dummy")  # For payload building only
    return provider._build_composition_payload(request)
