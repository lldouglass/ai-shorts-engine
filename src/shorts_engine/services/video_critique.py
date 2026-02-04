"""Video critique service for LLM-based video quality analysis."""

import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from shorts_engine.adapters.llm.base import LLMProvider, VisionMessage
from shorts_engine.adapters.llm.openai import OpenAIProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import settings
from shorts_engine.logging import get_logger
from shorts_engine.services.frame_extractor import ExtractedFrames, FrameExtractor

logger = get_logger(__name__)


@dataclass
class SceneVideoCritique:
    """Feedback for a single scene's video clip."""

    scene_number: int
    score: float
    feedback: str
    motion_quality: str  # e.g., "smooth", "jerky", "static"
    issues: list[str] = field(default_factory=list)


@dataclass
class VideoCritiqueResult:
    """Result of video critique analysis."""

    visual_coherence_score: float
    style_consistency_score: float
    motion_coherence_score: float  # NEW - movement quality
    temporal_consistency_score: float  # NEW - frame-to-frame consistency
    overall_passed: bool
    per_scene_feedback: list[SceneVideoCritique]
    cross_scene_feedback: str  # NEW - scene transitions
    improvement_suggestions: list[str]
    summary: str
    raw_response: dict[str, Any] | None = None
    model_used: str | None = None
    duration_seconds: float | None = None


class VideoCritiqueService:
    """Service for critiquing generated video clips using vision-capable LLMs.

    Sends extracted frames from video clips to a vision LLM for analysis of:
    - Visual coherence between scenes
    - Style consistency across the set
    - Motion coherence (natural/smooth movement)
    - Temporal consistency (frame-to-frame object consistency)
    - Scene transitions
    - Per-scene quality feedback
    - Improvement suggestions for regeneration
    """

    CRITIQUE_SYSTEM_PROMPT = """You are an expert visual quality analyst for AI-generated video clips used in short-form video content.

Your task is to analyze frames extracted from video clips representing sequential scenes in a short video.

For each scene, you'll receive multiple frames (typically first, middle, last) to assess quality.

Evaluate the following criteria and return a JSON object:

{
    "visual_coherence_score": 0.0-1.0,
    "style_consistency_score": 0.0-1.0,
    "motion_coherence_score": 0.0-1.0,
    "temporal_consistency_score": 0.0-1.0,
    "overall_passed": true/false,
    "per_scene_feedback": [
        {
            "scene_number": 1,
            "score": 0.0-1.0,
            "feedback": "Brief assessment of this scene",
            "motion_quality": "smooth|jerky|static",
            "issues": ["List of specific issues if any"]
        }
    ],
    "cross_scene_feedback": "Assessment of how scenes connect together",
    "improvement_suggestions": ["Specific suggestions for improving the videos"],
    "summary": "Brief overall assessment"
}

VISUAL COHERENCE (0.0-1.0):
Evaluates how well the scenes work together as a sequence:
- 0.0-0.3: Scenes feel disconnected, jarring transitions
- 0.4-0.6: Some coherence but noticeable inconsistencies
- 0.7-0.8: Good coherence, scenes flow well together
- 0.9-1.0: Excellent coherence, seamless visual narrative

STYLE CONSISTENCY (0.0-1.0):
Evaluates visual style uniformity across all clips:
- 0.0-0.3: Wildly different styles, looks like different sources
- 0.4-0.6: Some style drift but recognizable as same set
- 0.7-0.8: Good consistency with minor variations
- 0.9-1.0: Perfect style match across all clips

MOTION COHERENCE (0.0-1.0):
Evaluates the quality of movement within clips (compare frames within each scene):
- 0.0-0.3: Jarring, unnatural, or glitchy movement
- 0.4-0.6: Movement exists but feels mechanical or off
- 0.7-0.8: Natural movement with minor artifacts
- 0.9-1.0: Smooth, natural, believable motion

TEMPORAL CONSISTENCY (0.0-1.0):
Evaluates frame-to-frame consistency within each clip:
- 0.0-0.3: Objects/characters drastically change between frames
- 0.4-0.6: Some flickering or morphing of elements
- 0.7-0.8: Consistent with minor variations
- 0.9-1.0: Perfect temporal consistency

Key aspects to check:
1. Art style consistency (anime, realistic, etc.)
2. Color palette consistency
3. Character design consistency (if applicable)
4. Lighting direction consistency
5. Level of detail consistency
6. Movement quality and smoothness
7. Object/character persistence across frames
8. Scene transition compatibility

When providing improvement suggestions, be specific about:
- Which scenes need work
- What specific changes would help
- How to better maintain consistency
- Motion issues to address

Set overall_passed=true only if all scores are above their respective thresholds."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        frame_extractor: FrameExtractor | None = None,
    ) -> None:
        """Initialize the video critique service.

        Args:
            llm_provider: Optional LLM provider. If not provided, selects
                         a vision-capable provider based on configuration.
            frame_extractor: Optional frame extractor. If not provided,
                            creates a new instance.
        """
        self.llm = llm_provider or self._get_vision_provider()
        self.frame_extractor = frame_extractor or FrameExtractor()
        logger.info(
            "video_critique_service_initialized",
            provider=self.llm.name,
            supports_vision=self.llm.supports_vision,
        )

    def _get_vision_provider(self) -> LLMProvider:
        """Get a vision-capable LLM provider."""
        # Prefer Gemini for vision/video tasks (more cost-effective)
        if settings.google_api_key:
            from shorts_engine.adapters.llm.gemini import GeminiProvider

            return GeminiProvider(model=settings.gemini_critique_model)

        # Fall back to OpenAI GPT-4o
        if settings.openai_api_key:
            return OpenAIProvider(model="gpt-4o")

        # Fall back to stub for testing
        logger.warning("No vision-capable LLM configured, using stub provider")
        return StubLLMProvider()

    async def critique_scene_clips(
        self,
        clip_urls: list[str],
        scene_ids: list[UUID],
        style_preset: str,
        scene_descriptions: list[str] | None = None,
    ) -> VideoCritiqueResult:
        """Critique a set of scene video clips for quality metrics.

        Args:
            clip_urls: List of video URLs (or file paths) for each scene
            scene_ids: List of scene UUIDs corresponding to each clip
            style_preset: The style preset used for generation (for context)
            scene_descriptions: Optional list of scene descriptions for context

        Returns:
            VideoCritiqueResult with scores, feedback, and suggestions
        """
        if not self.llm.supports_vision:
            raise ValueError(f"Provider {self.llm.name} does not support vision")

        if not clip_urls:
            raise ValueError("No video clips provided for critique")

        if len(clip_urls) != len(scene_ids):
            raise ValueError("Number of clips must match number of scene IDs")

        start_time = time.time()
        logger.info(
            "video_critique_started",
            clip_count=len(clip_urls),
            style_preset=style_preset,
        )

        # Extract frames from all clips
        all_frames: list[ExtractedFrames] = []
        all_frame_uris: list[str] = []

        for url, scene_id in zip(clip_urls, scene_ids, strict=True):
            frames = await self.frame_extractor.extract_frames(
                video_url=url,
                scene_id=scene_id,
                num_frames=settings.ralph_frames_per_scene,
            )
            all_frames.append(frames)
            all_frame_uris.extend(frames.frame_data_uris)

        # Build the user message with context
        context_parts = [
            f"Style preset: {style_preset}",
            f"Number of scenes: {len(clip_urls)}",
            f"Frames per scene: {settings.ralph_frames_per_scene}",
        ]

        if scene_descriptions:
            context_parts.append("\nScene descriptions:")
            for i, desc in enumerate(scene_descriptions, 1):
                context_parts.append(f"  Scene {i}: {desc[:100]}...")

        context_parts.append(
            "\nAnalyze these frames from video clips for visual quality, "
            "style consistency, motion coherence, and temporal consistency."
        )
        context_parts.append(
            f"\nFrames are organized as {settings.ralph_frames_per_scene} frames per scene "
            "(first, middle, last showing movement progression)."
        )

        user_text = "\n".join(context_parts)

        # Create vision message with all frames
        messages = [
            VisionMessage(role="system", text=self.CRITIQUE_SYSTEM_PROMPT, image_urls=[]),
            VisionMessage(role="user", text=user_text, image_urls=all_frame_uris),
        ]

        try:
            response = await self.llm.complete_with_vision(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent evaluation
                max_tokens=2048,
                json_mode=True,
            )
            eval_data = json.loads(response.content)
        except Exception as e:
            logger.error("video_critique_llm_error", error=str(e))
            # On LLM error, return a passing result with warning
            return VideoCritiqueResult(
                visual_coherence_score=1.0,
                style_consistency_score=1.0,
                motion_coherence_score=1.0,
                temporal_consistency_score=1.0,
                overall_passed=True,
                per_scene_feedback=[],
                cross_scene_feedback="Critique skipped due to LLM error",
                improvement_suggestions=[],
                summary=f"Critique skipped due to LLM error: {e}",
                model_used=self.llm.name,
                duration_seconds=time.time() - start_time,
            )

        # Parse per-scene feedback
        per_scene = []
        for sf in eval_data.get("per_scene_feedback", []):
            per_scene.append(
                SceneVideoCritique(
                    scene_number=sf.get("scene_number", 0),
                    score=sf.get("score", 0.0),
                    feedback=sf.get("feedback", ""),
                    motion_quality=sf.get("motion_quality", "unknown"),
                    issues=sf.get("issues", []),
                )
            )

        # Extract scores
        visual_coherence = eval_data.get("visual_coherence_score", 0.0)
        style_consistency = eval_data.get("style_consistency_score", 0.0)
        motion_coherence = eval_data.get("motion_coherence_score", 0.0)
        temporal_consistency = eval_data.get("temporal_consistency_score", 0.0)

        # Determine pass/fail based on thresholds
        overall_passed = (
            visual_coherence >= settings.ralph_visual_coherence_threshold
            and style_consistency >= settings.ralph_style_consistency_threshold
            and motion_coherence >= settings.ralph_motion_coherence_threshold
            and temporal_consistency >= settings.ralph_temporal_consistency_threshold
        )

        duration = time.time() - start_time

        result = VideoCritiqueResult(
            visual_coherence_score=visual_coherence,
            style_consistency_score=style_consistency,
            motion_coherence_score=motion_coherence,
            temporal_consistency_score=temporal_consistency,
            overall_passed=overall_passed,
            per_scene_feedback=per_scene,
            cross_scene_feedback=eval_data.get("cross_scene_feedback", ""),
            improvement_suggestions=eval_data.get("improvement_suggestions", []),
            summary=eval_data.get("summary", ""),
            raw_response=eval_data,
            model_used=self.llm.name,
            duration_seconds=duration,
        )

        logger.info(
            "video_critique_completed",
            visual_coherence=visual_coherence,
            style_consistency=style_consistency,
            motion_coherence=motion_coherence,
            temporal_consistency=temporal_consistency,
            overall_passed=overall_passed,
            duration=duration,
        )

        return result


def get_video_critique_service() -> VideoCritiqueService:
    """Get a video critique service instance."""
    return VideoCritiqueService()
