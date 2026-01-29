"""Frame extraction service for video clip analysis."""

import base64
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
from PIL import Image

from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedFrames:
    """Result of frame extraction from a video clip."""

    scene_id: UUID
    frame_data_uris: list[str]  # base64 data URIs for vision LLM
    frame_timestamps: list[float]  # timestamps in seconds
    video_duration: float


class FrameExtractor:
    """Service for extracting key frames from video clips.

    Extracts frames at key points (start, middle, end) for analysis
    by vision-capable LLMs.
    """

    def __init__(self) -> None:
        """Initialize the frame extractor."""
        self._moviepy_available: bool | None = None

    def _check_moviepy(self) -> bool:
        """Check if moviepy is available."""
        if self._moviepy_available is None:
            try:
                import moviepy.editor  # noqa: F401

                self._moviepy_available = True
            except ImportError:
                self._moviepy_available = False
                logger.warning(
                    "moviepy_not_available",
                    message="Install moviepy for video frame extraction",
                )
        return self._moviepy_available

    async def extract_frames(
        self,
        video_url: str,
        scene_id: UUID,
        num_frames: int = 3,
    ) -> ExtractedFrames:
        """Extract key frames from a video clip.

        Extracts frames at evenly spaced intervals (start, middle, end by default).

        Args:
            video_url: URL or file path to the video clip
            scene_id: UUID of the scene this video belongs to
            num_frames: Number of frames to extract (default 3: start, middle, end)

        Returns:
            ExtractedFrames with base64 data URIs for each frame
        """
        logger.info(
            "frame_extraction_started",
            scene_id=str(scene_id),
            video_url=video_url[:100],
            num_frames=num_frames,
        )

        if not self._check_moviepy():
            # Return stub result when moviepy is not available
            return self._create_stub_frames(scene_id, num_frames)

        # Download video to temp file if it's a URL
        video_path = await self._get_video_path(video_url)

        try:
            frames, timestamps, duration = self._extract_frames_from_file(video_path, num_frames)

            # Convert frames to base64 data URIs
            frame_data_uris = [self._frame_to_data_uri(frame) for frame in frames]

            logger.info(
                "frame_extraction_completed",
                scene_id=str(scene_id),
                frame_count=len(frame_data_uris),
                video_duration=duration,
            )

            return ExtractedFrames(
                scene_id=scene_id,
                frame_data_uris=frame_data_uris,
                frame_timestamps=timestamps,
                video_duration=duration,
            )
        finally:
            # Clean up temp file if we downloaded it
            if video_path != video_url and Path(video_path).exists():
                try:
                    Path(video_path).unlink()
                except Exception as e:
                    logger.warning("temp_file_cleanup_failed", error=str(e))

    async def _get_video_path(self, video_url: str) -> str:
        """Get local file path for video, downloading if necessary.

        Args:
            video_url: URL or file path to video

        Returns:
            Local file path to the video
        """
        # Check if it's already a local file
        if video_url.startswith("file://"):
            return video_url[7:]  # Strip file:// prefix

        if not video_url.startswith(("http://", "https://")):
            # Assume it's a local path
            return video_url

        # Download to temp file
        logger.debug("downloading_video_for_extraction", url=video_url[:100])

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(video_url)
            response.raise_for_status()

            # Write to temp file
            suffix = ".mp4"
            if "content-type" in response.headers:
                content_type = response.headers["content-type"]
                if "webm" in content_type:
                    suffix = ".webm"
                elif "avi" in content_type:
                    suffix = ".avi"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(response.content)
                return temp_file.name

    def _extract_frames_from_file(
        self,
        video_path: str,
        num_frames: int,
    ) -> tuple[list[Image.Image], list[float], float]:
        """Extract frames from a video file using moviepy.

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract

        Returns:
            Tuple of (frames as PIL Images, timestamps, video duration)
        """
        from moviepy.editor import VideoFileClip

        frames: list[Image.Image] = []
        timestamps: list[float] = []

        with VideoFileClip(video_path) as clip:
            duration = clip.duration

            # Calculate timestamps for frame extraction
            if num_frames == 1:
                extract_times = [duration / 2]
            elif num_frames == 2:
                extract_times = [0.1, duration - 0.1]
            else:
                # Evenly spaced, avoiding exact start/end
                step = duration / (num_frames + 1)
                extract_times = [step * (i + 1) for i in range(num_frames)]

            # Extract frames at each timestamp
            for timestamp in extract_times:
                # Clamp timestamp to valid range
                t = max(0.0, min(timestamp, duration - 0.01))

                # Get frame as numpy array and convert to PIL Image
                frame_array = clip.get_frame(t)
                frame = Image.fromarray(frame_array)
                frames.append(frame)
                timestamps.append(t)

        return frames, timestamps, duration

    def _frame_to_data_uri(self, frame: Image.Image) -> str:
        """Convert a PIL Image to a base64 data URI.

        Args:
            frame: PIL Image to convert

        Returns:
            Base64 data URI string
        """
        # Resize if too large (for efficiency)
        max_dim = 1024
        if frame.width > max_dim or frame.height > max_dim:
            frame.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        # Convert to JPEG for smaller size
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        # Encode to base64
        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"

    def _create_stub_frames(
        self,
        scene_id: UUID,
        num_frames: int,
    ) -> ExtractedFrames:
        """Create stub frames when moviepy is not available.

        Args:
            scene_id: UUID of the scene
            num_frames: Number of frames to simulate

        Returns:
            ExtractedFrames with placeholder data
        """
        # Create a simple placeholder image
        placeholder = Image.new("RGB", (640, 360), color=(50, 50, 50))

        frame_data_uris = [self._frame_to_data_uri(placeholder) for _ in range(num_frames)]
        timestamps = [i * 2.0 for i in range(num_frames)]
        duration = num_frames * 2.0

        return ExtractedFrames(
            scene_id=scene_id,
            frame_data_uris=frame_data_uris,
            frame_timestamps=timestamps,
            video_duration=duration,
        )


def get_frame_extractor() -> FrameExtractor:
    """Get a frame extractor service instance."""
    return FrameExtractor()
