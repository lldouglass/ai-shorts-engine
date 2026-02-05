"""Utilities for extracting frames from video clips."""

import subprocess
from pathlib import Path

from shorts_engine.logging import get_logger

logger = get_logger(__name__)


def extract_last_frame(video_path: Path, output_path: Path | None = None) -> bytes:
    """Extract the last frame from a video as JPEG bytes.

    Tries ffmpeg first, falls back to moviepy if ffmpeg is not available.

    Args:
        video_path: Path to the video file.
        output_path: Optional path to save the frame. If None, uses a temp path.

    Returns:
        JPEG image bytes of the last frame.

    Raises:
        RuntimeError: If frame extraction fails with both methods.
    """
    if output_path is None:
        output_path = video_path.with_suffix(".last_frame.jpg")

    # Try ffmpeg first
    try:
        return _extract_last_frame_ffmpeg(video_path, output_path)
    except (FileNotFoundError, RuntimeError) as e:
        logger.debug(
            "ffmpeg_not_available_trying_moviepy",
            error=str(e),
        )
        # Fall back to moviepy
        return extract_last_frame_moviepy(video_path, output_path)


def _extract_last_frame_ffmpeg(video_path: Path, output_path: Path) -> bytes:
    """Extract last frame using ffmpeg."""
    # First, get video duration using ffprobe
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    duration_result = subprocess.run(
        probe_cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(duration_result.stdout.strip())

    logger.debug(
        "frame_extraction_duration",
        video_path=str(video_path),
        duration=duration,
    )

    # Extract frame at duration - 0.1s (to avoid edge issues)
    seek_time = max(0, duration - 0.1)
    extract_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss",
        str(seek_time),  # Seek to near end
        "-i",
        str(video_path),
        "-vframes",
        "1",  # Extract 1 frame
        "-q:v",
        "2",  # High quality JPEG
        str(output_path),
    ]
    subprocess.run(
        extract_cmd,
        capture_output=True,
        check=True,
    )

    # Read the frame bytes
    frame_bytes = output_path.read_bytes()

    logger.info(
        "frame_extraction_success",
        video_path=str(video_path),
        output_path=str(output_path),
        frame_size=len(frame_bytes),
    )

    return frame_bytes


def extract_last_frame_moviepy(video_path: Path, output_path: Path | None = None) -> bytes:
    """Extract last frame using MoviePy (fallback if ffmpeg not available).

    Args:
        video_path: Path to the video file.
        output_path: Optional path to save the frame.

    Returns:
        JPEG image bytes of the last frame.
    """
    from moviepy import VideoFileClip

    if output_path is None:
        output_path = video_path.with_suffix(".last_frame.jpg")

    try:
        clip = VideoFileClip(str(video_path))
        # Get frame at 0.1s before end
        frame_time = max(0, clip.duration - 0.1)
        clip.save_frame(str(output_path), t=frame_time)
        clip.close()

        frame_bytes = output_path.read_bytes()

        logger.info(
            "frame_extraction_moviepy_success",
            video_path=str(video_path),
            frame_size=len(frame_bytes),
        )

        return frame_bytes

    except Exception as e:
        logger.error(
            "frame_extraction_moviepy_error",
            video_path=str(video_path),
            error=str(e),
        )
        raise RuntimeError(f"MoviePy frame extraction failed: {e}") from e
