"""Direct renderer for approved two-scene preview formats."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from shorts_engine.adapters.renderer.creatomate import (
    ImageCompositionRequest,
    ImageSceneClip,
    MotionParams,
)
from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer
from shorts_engine.adapters.voiceover.base import VoiceoverRequest
from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider
from shorts_engine.config import settings
from shorts_engine.utils import run_async

SUPPORTED_APPROVED_BRANDS = {"car", "moatifi"}
DEFAULT_VOICE_BY_BRAND = {
    "car": "en-US-AndrewNeural",
    "moatifi": "en-US-AndrewNeural",
}


@dataclass
class ApprovedPreviewScene:
    """Single approved-preview scene input."""

    image_path: Path
    line: str


@dataclass
class ApprovedPreviewRenderResult:
    """Result for a direct approved-preview render."""

    final_video_path: Path
    voiceover_path: Path
    scene_durations: list[float]
    total_duration_seconds: float


def validate_approved_preview_inputs(
    brand: str,
    scene_images: list[str],
    scene_lines: list[str],
) -> list[ApprovedPreviewScene]:
    """Validate CLI inputs for approved-preview rendering."""
    normalized_brand = brand.strip().lower()
    if normalized_brand not in SUPPORTED_APPROVED_BRANDS:
        allowed = ", ".join(sorted(SUPPORTED_APPROVED_BRANDS))
        raise ValueError(f"Unsupported brand '{brand}'. Expected one of: {allowed}")

    if len(scene_images) != 2:
        raise ValueError("Approved preview render requires exactly 2 --scene-image values")
    if len(scene_lines) != 2:
        raise ValueError("Approved preview render requires exactly 2 --scene-line values")

    scenes: list[ApprovedPreviewScene] = []
    for image_str, line in zip(scene_images, scene_lines, strict=True):
        image_path = Path(image_str).expanduser()
        if not image_path.exists():
            raise ValueError(f"Scene image not found: {image_path}")
        cleaned_line = _clean_line(line)
        if not cleaned_line:
            raise ValueError("Scene lines must be non-empty")
        scenes.append(ApprovedPreviewScene(image_path=image_path, line=cleaned_line))

    return scenes


def render_approved_preview(
    brand: str,
    output_name: str,
    scenes: list[ApprovedPreviewScene],
    voice_id: str | None = None,
    speed: float = 0.95,
    background_music_url: str | None = None,
    background_music_volume: float | None = None,
    output_dir: Path | None = None,
) -> ApprovedPreviewRenderResult:
    """Render a two-scene approved preview into one final vertical MP4."""
    normalized_brand = brand.strip().lower()
    if normalized_brand not in SUPPORTED_APPROVED_BRANDS:
        allowed = ", ".join(sorted(SUPPORTED_APPROVED_BRANDS))
        raise ValueError(f"Unsupported brand '{brand}'. Expected one of: {allowed}")
    if len(scenes) != 2:
        raise ValueError("Approved preview render currently supports exactly 2 scenes")

    output_root = output_dir or Path("output/approved_previews")
    output_root.mkdir(parents=True, exist_ok=True)

    voiceover_provider = EdgeTTSProvider()
    narration_text = " ".join(scene.line for scene in scenes)
    voice = voice_id or DEFAULT_VOICE_BY_BRAND[normalized_brand]

    voiceover = run_async(
        voiceover_provider.generate(
            VoiceoverRequest(
                text=narration_text,
                voice_id=voice,
                speed=speed,
            )
        )
    )
    if not voiceover.success or not voiceover.audio_data:
        raise RuntimeError(voiceover.error_message or "Voiceover generation failed")

    voiceover_path = output_root / f"{output_name}_voiceover.mp3"
    voiceover_path.write_bytes(voiceover.audio_data)

    total_duration = float(voiceover.duration_seconds or 0.0)
    scene_durations = _estimate_scene_durations(
        [scene.line for scene in scenes],
        voiceover.metadata.get("word_boundaries") or [],
        total_duration,
    )

    effective_music_url, effective_music_volume = _resolve_background_music(
        background_music_url,
        background_music_volume,
    )

    image_clips = [
        ImageSceneClip(
            image_url=str(scene.image_path),
            duration_seconds=scene_durations[index],
            motion=_default_motion(index),
            caption_text=scene.line,
            scene_number=index + 1,
            transition="cut",
            transition_duration=0.0,
        )
        for index, scene in enumerate(scenes)
    ]

    renderer = MoviePyRenderer(output_dir=output_root / "tmp")
    render_result = run_async(
        renderer.render_image_composition(
            ImageCompositionRequest(
                images=image_clips,
                voiceover_url=str(voiceover_path),
                background_music_url=effective_music_url,
                background_music_volume=effective_music_volume,
                timed_captions=None,
                width=1080,
                height=1920,
                fps=30,
            )
        )
    )
    if not render_result.success or not render_result.output_path:
        raise RuntimeError(render_result.error_message or "Render failed")

    final_video_path = output_root / f"{output_name}.mp4"
    shutil.copy2(render_result.output_path, final_video_path)

    return ApprovedPreviewRenderResult(
        final_video_path=final_video_path,
        voiceover_path=voiceover_path,
        scene_durations=scene_durations,
        total_duration_seconds=float(render_result.duration_seconds or total_duration),
    )


def _clean_line(text: str) -> str:
    """Normalize scene lines for captions and narration."""
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def _estimate_scene_durations(
    scene_lines: list[str],
    word_boundaries: list[dict[str, Any]],
    total_duration_seconds: float,
) -> list[float]:
    """Estimate per-scene durations from voice timing, falling back to word ratios."""
    minimum = 1.0
    if len(scene_lines) != 2:
        raise ValueError("Expected exactly 2 scene lines")

    tokens = [
        str(item.get("text") or "").strip() for item in word_boundaries if item.get("text")
    ]
    if not tokens:
        return _fallback_scene_durations(scene_lines, total_duration_seconds, minimum)

    scene_word_counts = [max(1, len(_clean_line(line).split())) for line in scene_lines]
    split_index = min(scene_word_counts[0], len(tokens) - 1)

    if split_index <= 0 or split_index >= len(word_boundaries):
        return _fallback_scene_durations(scene_lines, total_duration_seconds, minimum)

    first_end = float(word_boundaries[split_index - 1].get("end_seconds") or 0.0)
    total_duration = max(total_duration_seconds, first_end + minimum)
    second_duration = max(minimum, total_duration - first_end)
    first_duration = max(minimum, total_duration - second_duration)

    return [first_duration, second_duration]


def _fallback_scene_durations(
    scene_lines: list[str],
    total_duration_seconds: float,
    minimum: float,
) -> list[float]:
    """Split total duration proportionally by scene word counts."""
    total_duration = max(total_duration_seconds, minimum * len(scene_lines))
    weights = [max(1, len(_clean_line(line).split())) for line in scene_lines]
    total_weight = sum(weights)

    first_duration = max(minimum, total_duration * weights[0] / total_weight)
    second_duration = max(minimum, total_duration - first_duration)
    return [first_duration, second_duration]


def _default_motion(index: int) -> MotionParams:
    """Slightly vary Ken Burns motion across the two approved scenes."""
    if index == 0:
        return MotionParams(
            zoom_start=1.0,
            zoom_end=1.06,
            pan_x_start=0.0,
            pan_x_end=-0.04,
        )
    return MotionParams(
        zoom_start=1.02,
        zoom_end=1.08,
        pan_x_start=0.03,
        pan_x_end=0.0,
    )


def _resolve_background_music(
    background_music_url: str | None,
    background_music_volume: float | None,
) -> tuple[str | None, float]:
    """Resolve optional background music using existing app settings."""
    effective_volume = (
        float(background_music_volume)
        if background_music_volume is not None
        else float(settings.background_music_default_volume)
    )

    if background_music_url:
        return background_music_url, effective_volume

    if not settings.background_music_enabled or not settings.background_music_default_url:
        return None, effective_volume

    candidate = Path(settings.background_music_default_url)
    if candidate.exists():
        return str(candidate), effective_volume

    repo_candidate = Path.cwd() / candidate
    if repo_candidate.exists():
        return str(repo_candidate), effective_volume

    return None, effective_volume


def build_default_output_name(brand: str) -> str:
    """Create a stable default output name for manual runs."""
    run_id = uuid4().hex[:8]
    return f"{brand.strip().lower()}_approved_preview_{run_id}"
