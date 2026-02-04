"""MoviePy local video renderer.

Uses MoviePy 2.x (which bundles ffmpeg via imageio_ffmpeg) for local video
composition and rendering without requiring external cloud services.
"""

import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from shorts_engine.adapters.renderer.base import RendererProvider, RenderRequest, RenderResult
from shorts_engine.adapters.renderer.creatomate import (
    CreatomateRenderRequest,
    ImageCompositionRequest,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

# MoviePy 2.x imports
try:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class MoviePyRenderer(RendererProvider):
    """Local video renderer using MoviePy (bundled ffmpeg).

    This renderer provides local video composition capabilities without
    requiring cloud services like Creatomate. It supports:
    - Concatenating video clips
    - Adding voiceover and background music
    - Ken Burns effects on images
    - Burned-in captions
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        fps: int = 30,
        codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> None:
        """Initialize the MoviePy renderer.

        Args:
            output_dir: Directory for output files. Uses temp dir if None.
            fps: Output video frame rate.
            codec: Video codec for encoding.
            audio_codec: Audio codec for encoding.
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "shorts_engine"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.audio_codec = audio_codec

    @property
    def name(self) -> str:
        return "moviepy"

    async def render(self, request: RenderRequest) -> RenderResult:
        """Render using basic RenderRequest (transcode operation).

        Args:
            request: Basic render request with video data.

        Returns:
            RenderResult with output path or error.
        """
        try:
            if not MOVIEPY_AVAILABLE:
                return RenderResult(success=False, error_message="MoviePy not available")

            # Write input video data to temp file
            input_path = self.output_dir / f"input_{uuid4().hex}.mp4"
            input_path.write_bytes(request.video_data)

            # Load and process
            clip = VideoFileClip(str(input_path))

            # Parse resolution
            width, height = 1080, 1920
            if request.resolution:
                parts = request.resolution.split("x")
                if len(parts) == 2:
                    width, height = int(parts[0]), int(parts[1])

            # Resize if needed
            if clip.size != (width, height):
                clip = clip.resized(new_size=(width, height))

            # Add audio if provided
            if request.audio_track:
                audio_path = self.output_dir / f"audio_{uuid4().hex}.mp3"
                audio_path.write_bytes(request.audio_track)
                audio = AudioFileClip(str(audio_path))
                clip = clip.with_audio(audio)

            # Store duration before closing
            duration = clip.duration

            # Output
            output_path = self.output_dir / f"output_{uuid4().hex}.mp4"
            clip.write_videofile(
                str(output_path),
                fps=request.fps,
                codec=self.codec,
                audio_codec=self.audio_codec,
                logger=None,  # Suppress moviepy progress output
            )

            # Cleanup
            clip.close()
            input_path.unlink(missing_ok=True)

            return RenderResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size,
                duration_seconds=duration,
                metadata={"provider": self.name, "url": f"file://{output_path}"},
            )

        except Exception as e:
            logger.error("moviepy_render_error", error=str(e))
            return RenderResult(success=False, error_message=str(e))

    async def render_composition(
        self,
        request: CreatomateRenderRequest,
    ) -> RenderResult:
        """Render a composition by concatenating video clips.

        Args:
            request: Extended render request with scene clips.

        Returns:
            RenderResult with output path or error.
        """
        try:
            if not MOVIEPY_AVAILABLE:
                return RenderResult(success=False, error_message="MoviePy not available")

            logger.info(
                "moviepy_composition_started",
                scene_count=len(request.scenes),
                has_voiceover=request.voiceover_url is not None,
            )

            # Download all video clips
            video_clips = []
            temp_files: list[Path] = []

            for i, scene in enumerate(request.scenes):
                logger.debug("downloading_scene_clip", scene_number=i + 1)
                clip_path = await self._download_file(scene.video_url, f"clip_{i}")
                temp_files.append(clip_path)

                clip = VideoFileClip(str(clip_path))

                # Trim to specified duration if needed
                if clip.duration > scene.duration_seconds:
                    clip = clip.subclipped(0, scene.duration_seconds)

                # Skip captions for now (font issues on Windows)
                # Can be added back with proper font handling

                video_clips.append(clip)

            # Concatenate all clips
            final_video = concatenate_videoclips(video_clips, method="compose")

            # Handle audio tracks
            audio_clips = []

            # Add voiceover
            if request.voiceover_url:
                voiceover_path = await self._download_file(request.voiceover_url, "voiceover")
                temp_files.append(voiceover_path)
                voiceover = AudioFileClip(str(voiceover_path))

                # Handle voiceover duration mismatch with video
                if voiceover.duration > final_video.duration:
                    # Voiceover longer than video - clip it
                    logger.warning(
                        "voiceover_clipped",
                        voiceover_duration=voiceover.duration,
                        video_duration=final_video.duration,
                    )
                    voiceover = voiceover.subclipped(0, final_video.duration)
                elif voiceover.duration < final_video.duration - 5:
                    # Voiceover significantly shorter (>5s gap) - log warning
                    # The remaining video will have silence, which may be intentional
                    logger.warning(
                        "voiceover_shorter_than_video",
                        voiceover_duration=voiceover.duration,
                        video_duration=final_video.duration,
                        gap_seconds=final_video.duration - voiceover.duration,
                    )

                audio_clips.append(voiceover)

            # Add background music (reduced volume)
            if request.background_music_url:
                music_path = await self._download_file(request.background_music_url, "music")
                temp_files.append(music_path)
                music = AudioFileClip(str(music_path))
                # Loop if shorter than video
                if music.duration < final_video.duration:
                    music = music.with_effects(
                        [lambda clip: clip.loop(duration=final_video.duration)]
                    )
                else:
                    music = music.subclipped(0, final_video.duration)
                music = music.with_volume_scaled(request.background_music_volume)
                audio_clips.append(music)

            # Composite audio tracks
            if audio_clips:
                final_audio = CompositeAudioClip(audio_clips)
                final_video = final_video.with_audio(final_audio)

            # Resize to target dimensions
            final_video = final_video.resized(new_size=(request.width, request.height))

            # Store duration before writing (close may clear it)
            duration = final_video.duration

            # Write output
            output_path = self.output_dir / f"composition_{uuid4().hex}.mp4"
            final_video.write_videofile(
                str(output_path),
                fps=request.fps,
                codec=self.codec,
                audio_codec=self.audio_codec,
                logger=None,
            )

            # Cleanup
            final_video.close()
            for clip in video_clips:
                clip.close()
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)

            logger.info(
                "moviepy_composition_completed",
                output_path=str(output_path),
                duration=duration,
            )

            return RenderResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size,
                duration_seconds=duration,
                metadata={"provider": self.name, "url": f"file://{output_path}"},
            )

        except Exception as e:
            logger.error("moviepy_composition_error", error=str(e))
            return RenderResult(success=False, error_message=str(e))

    async def render_image_composition(
        self,
        request: ImageCompositionRequest,
    ) -> RenderResult:
        """Render image-based composition with Ken Burns effects.

        Args:
            request: Image composition request with motion parameters.

        Returns:
            RenderResult with output path or error.
        """
        try:
            if not MOVIEPY_AVAILABLE:
                return RenderResult(success=False, error_message="MoviePy not available")

            logger.info(
                "moviepy_image_composition_started",
                image_count=len(request.images),
                has_voiceover=request.voiceover_url is not None,
            )

            video_clips = []
            temp_files: list[Path] = []

            for i, img_scene in enumerate(request.images):
                logger.debug("processing_image", image_number=i + 1)

                # Download image
                img_path = await self._download_file(img_scene.image_url, f"img_{i}")
                temp_files.append(img_path)

                # Create image clip with Ken Burns effect
                clip = self._create_ken_burns_clip(
                    str(img_path),
                    img_scene.duration_seconds,
                    request.width,
                    request.height,
                    img_scene.motion.zoom_start,
                    img_scene.motion.zoom_end,
                    img_scene.motion.pan_x_start,
                    img_scene.motion.pan_x_end,
                    img_scene.motion.pan_y_start,
                    img_scene.motion.pan_y_end,
                )

                # Skip captions for now (font issues on Windows)

                video_clips.append(clip)

            # Concatenate with transitions
            final_video = concatenate_videoclips(video_clips, method="compose")

            # Handle audio tracks
            audio_clips = []

            if request.voiceover_url:
                voiceover_path = await self._download_file(request.voiceover_url, "voiceover")
                temp_files.append(voiceover_path)
                voiceover = AudioFileClip(str(voiceover_path))

                # Handle voiceover duration mismatch with video
                if voiceover.duration > final_video.duration:
                    logger.warning(
                        "voiceover_clipped",
                        voiceover_duration=voiceover.duration,
                        video_duration=final_video.duration,
                    )
                    voiceover = voiceover.subclipped(0, final_video.duration)
                elif voiceover.duration < final_video.duration - 5:
                    logger.warning(
                        "voiceover_shorter_than_video",
                        voiceover_duration=voiceover.duration,
                        video_duration=final_video.duration,
                        gap_seconds=final_video.duration - voiceover.duration,
                    )

                audio_clips.append(voiceover)

            if request.background_music_url:
                music_path = await self._download_file(request.background_music_url, "music")
                temp_files.append(music_path)
                music = AudioFileClip(str(music_path))
                if music.duration < final_video.duration:
                    pass  # Skip looping for now
                else:
                    music = music.subclipped(0, final_video.duration)
                music = music.with_volume_scaled(request.background_music_volume)
                audio_clips.append(music)

            if audio_clips:
                final_audio = CompositeAudioClip(audio_clips)
                final_video = final_video.with_audio(final_audio)

            # Store duration before writing
            duration = final_video.duration

            # Write output
            output_path = self.output_dir / f"image_composition_{uuid4().hex}.mp4"
            final_video.write_videofile(
                str(output_path),
                fps=request.fps,
                codec=self.codec,
                audio_codec=self.audio_codec,
                logger=None,
            )

            # Cleanup
            final_video.close()
            for clip in video_clips:
                clip.close()
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)

            logger.info(
                "moviepy_image_composition_completed",
                output_path=str(output_path),
                duration=duration,
            )

            return RenderResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size,
                duration_seconds=duration,
                metadata={"provider": self.name, "url": f"file://{output_path}"},
            )

        except Exception as e:
            logger.error("moviepy_image_composition_error", error=str(e))
            return RenderResult(success=False, error_message=str(e))

    def _create_ken_burns_clip(
        self,
        image_path: str,
        duration: float,
        width: int,
        height: int,
        zoom_start: float,
        zoom_end: float,
        pan_x_start: float,
        pan_x_end: float,
        pan_y_start: float,
        pan_y_end: float,
    ) -> Any:
        """Create an image clip with Ken Burns zoom/pan effect.

        Args:
            image_path: Path to source image.
            duration: Clip duration in seconds.
            width: Output width.
            height: Output height.
            zoom_start: Starting zoom level (1.0 = no zoom).
            zoom_end: Ending zoom level.
            pan_x_start: Starting horizontal pan (-0.5 to 0.5).
            pan_x_end: Ending horizontal pan.
            pan_y_start: Starting vertical pan (-0.5 to 0.5).
            pan_y_end: Ending vertical pan.

        Returns:
            VideoClip with Ken Burns effect applied.
        """
        # Load image and set duration
        img_clip = ImageClip(image_path).with_duration(duration)

        # Get original dimensions
        orig_w, orig_h = img_clip.size

        # Calculate the zoom and pan at each time point
        def make_frame(get_frame: Any) -> Any:
            def new_frame(t: float) -> Any:
                # Interpolate zoom
                progress = t / duration
                zoom = zoom_start + (zoom_end - zoom_start) * progress

                # Interpolate pan
                pan_x = pan_x_start + (pan_x_end - pan_x_start) * progress
                pan_y = pan_y_start + (pan_y_end - pan_y_start) * progress

                # Calculate crop region
                crop_w = int(orig_w / zoom)
                crop_h = int(orig_h / zoom)

                # Center with pan offset
                center_x = orig_w / 2 + pan_x * orig_w
                center_y = orig_h / 2 + pan_y * orig_h

                # Crop coordinates
                x1 = int(max(0, center_x - crop_w / 2))
                y1 = int(max(0, center_y - crop_h / 2))
                x2 = int(min(orig_w, x1 + crop_w))
                y2 = int(min(orig_h, y1 + crop_h))

                # Get frame and crop
                frame = get_frame(t)
                cropped = frame[y1:y2, x1:x2]

                # Resize to output dimensions
                import numpy as np
                from PIL import Image

                pil_img = Image.fromarray(cropped)
                pil_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
                return np.array(pil_img)

            return new_frame

        # Apply the Ken Burns transformation using transform
        return img_clip.transform(make_frame)

    def _create_caption_clip(
        self,
        text: str,
        duration: float,
        width: int,
    ) -> Any:
        """Create a caption text clip.

        Note: Currently disabled due to font issues on Windows.
        To enable, install ImageMagick and ensure fonts are available.

        Args:
            text: Caption text.
            duration: Duration in seconds.
            width: Video width for text sizing.

        Returns:
            TextClip with caption styling or None if unavailable.
        """
        # TextClip requires ImageMagick which isn't always available
        # Return None and skip captions for now
        return None

    async def _download_file(self, url: str, prefix: str) -> Path:
        """Download a file from URL to temp directory.

        Args:
            url: Source URL.
            prefix: Filename prefix.

        Returns:
            Path to downloaded file.
        """
        # Handle file:// URLs
        if url.startswith("file://"):
            return Path(url[7:])

        # Determine extension from URL
        ext = ".mp4"
        if ".mp3" in url or "audio" in url.lower():
            ext = ".mp3"
        elif ".png" in url:
            ext = ".png"
        elif ".jpg" in url or ".jpeg" in url:
            ext = ".jpg"
        elif ".webp" in url:
            ext = ".webp"

        output_path = self.output_dir / f"{prefix}_{uuid4().hex}{ext}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)

        return output_path

    async def get_video_info(self, video_path: Path) -> dict[str, Any]:
        """Get information about a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video metadata.
        """
        try:
            if not MOVIEPY_AVAILABLE:
                return {"error": "MoviePy not available", "path": str(video_path)}

            clip = VideoFileClip(str(video_path))
            info = {
                "duration": clip.duration,
                "size": clip.size,
                "fps": clip.fps,
                "path": str(video_path),
            }
            clip.close()
            return info
        except Exception as e:
            return {"error": str(e), "path": str(video_path)}

    async def health_check(self) -> bool:
        """Check if MoviePy and ffmpeg are available.

        Returns:
            True if MoviePy can be imported and ffmpeg is available.
        """
        try:
            if not MOVIEPY_AVAILABLE:
                return False

            import imageio_ffmpeg

            # Check ffmpeg is accessible
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            return ffmpeg_path is not None
        except Exception as e:
            logger.error("moviepy_health_check_failed", error=str(e))
            return False
