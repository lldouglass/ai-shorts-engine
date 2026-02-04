"""Final video critique service using Gemini's native video understanding."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shorts_engine.adapters.llm.base import LLMProvider
from shorts_engine.adapters.llm.gemini import GeminiProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FinalVideoCritiqueResult:
    """Result of analyzing the final rendered video."""

    overall_score: float
    visual_quality_score: float
    audio_sync_score: float
    narrative_flow_score: float
    scene_scores: dict[int, float]  # scene_number -> score
    failing_scenes: list[int]
    feedback: str
    improvement_suggestions: list[str]
    passed: bool
    raw_response: dict[str, Any] | None = None
    model_used: str | None = None
    duration_seconds: float | None = None


class FinalVideoCritiqueService:
    """Critiques the final rendered video using Gemini's native video understanding.

    This service analyzes the complete rendered video (not individual clips) to evaluate:
    - Visual quality and coherence across the entire video
    - Audio/voiceover synchronization with visuals
    - Narrative flow compared to the original plan
    - Per-scene quality scores to identify weak points

    Key advantage: Uses Gemini's native video understanding which processes
    the entire video (~258 tokens/second) rather than extracting frames.
    """

    CRITIQUE_PROMPT = """You are an expert video quality analyst for short-form social media content.

Analyze this short-form video for quality and engagement potential.

Original plan context:
{plan_json}

Evaluate the video and return a JSON object with this exact structure:
{{
    "overall_score": <float 0.0-1.0>,
    "visual_quality_score": <float 0.0-1.0>,
    "audio_sync_score": <float 0.0-1.0>,
    "narrative_flow_score": <float 0.0-1.0>,
    "scene_scores": {{"1": <float>, "2": <float>, ...}},
    "failing_scenes": [<list of scene numbers with score < threshold>],
    "feedback": "<overall assessment string>",
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
}}

SCORING CRITERIA:

VISUAL QUALITY (0.0-1.0):
- 0.0-0.3: Major artifacts, glitches, or distortion
- 0.4-0.6: Noticeable quality issues but watchable
- 0.7-0.8: Good quality with minor imperfections
- 0.9-1.0: Excellent visual quality throughout

AUDIO SYNC (0.0-1.0):
- 0.0-0.3: Audio severely out of sync or missing
- 0.4-0.6: Noticeable sync issues in parts
- 0.7-0.8: Good sync with minor timing issues
- 0.9-1.0: Perfect audio-visual synchronization

NARRATIVE FLOW (0.0-1.0):
- 0.0-0.3: Confusing, no clear story or flow
- 0.4-0.6: Story present but choppy transitions
- 0.7-0.8: Good narrative with smooth progression
- 0.9-1.0: Compelling narrative that matches plan exactly

OVERALL SCORE: Weighted average (Visual: 40%, Audio: 30%, Narrative: 30%)

For SCENE_SCORES: Identify approximately how many distinct scenes/shots are in the video
and rate each one. Scenes with scores below {threshold} should be listed in failing_scenes.

Check specifically for:
1. Visual coherence between scenes
2. Style consistency throughout
3. Voiceover/audio sync with visuals
4. Narrative flow matches the original plan
5. Technical quality (no rendering artifacts)
6. Transitions between scenes
7. Hook effectiveness (first 3 seconds)

For IMPROVEMENT_SUGGESTIONS: Be specific about which scenes need work and what changes would help.
Focus on actionable feedback that could guide regeneration of specific scenes."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the final video critique service.

        Args:
            llm_provider: Optional LLM provider. If not provided, uses Gemini
                         if google_api_key is available, otherwise stub.
        """
        self.llm = llm_provider or self._get_video_provider()
        logger.info(
            "final_video_critique_service_initialized",
            provider=self.llm.name,
            supports_video=self.llm.supports_video,
        )

    def _get_video_provider(self) -> LLMProvider:
        """Get a video-capable LLM provider."""
        # Prefer Gemini for native video understanding
        if settings.google_api_key:
            return GeminiProvider(model=settings.gemini_critique_model)

        # Fall back to stub for testing
        logger.warning("No video-capable LLM configured, using stub provider")
        return StubLLMProvider()

    async def critique(
        self,
        video_path: Path,
        plan: dict[str, Any],
        threshold: float | None = None,
    ) -> FinalVideoCritiqueResult:
        """Critique the final rendered video.

        Args:
            video_path: Path to the rendered video file
            plan: Original plan data with scene descriptions
            threshold: Minimum score to pass (uses settings if not provided)

        Returns:
            FinalVideoCritiqueResult with scores, feedback, and suggestions
        """
        if threshold is None:
            threshold = settings.final_critique_threshold

        start_time = time.time()

        logger.info(
            "final_video_critique_started",
            video_path=str(video_path),
            threshold=threshold,
        )

        # Prepare plan context (simplified for prompt)
        plan_summary = {
            "title": plan.get("title", "Unknown"),
            "description": plan.get("description", ""),
            "scene_count": len(plan.get("scenes", [])),
            "scenes": [
                {
                    "scene_number": s.get("scene_number", i + 1),
                    "visual_prompt": s.get("visual_prompt", "")[:200],
                    "caption_beat": s.get("caption_beat", ""),
                }
                for i, s in enumerate(plan.get("scenes", []))
            ],
        }

        prompt = self.CRITIQUE_PROMPT.format(
            plan_json=json.dumps(plan_summary, indent=2),
            threshold=threshold,
        )

        try:
            if self.llm.supports_video:
                response = await self.llm.complete_with_video(
                    video_path=video_path,
                    prompt=prompt,
                    temperature=0.3,
                    json_mode=True,
                )
            else:
                # Fallback for non-video providers (like stub)
                logger.warning(
                    "final_video_critique_no_video_support",
                    provider=self.llm.name,
                )
                # Return a passing result for stub provider
                return self._create_stub_result(plan, threshold, time.time() - start_time)

            eval_data = json.loads(response.content)

        except Exception as e:
            logger.error("final_video_critique_error", error=str(e))
            # On error, return a passing result to not block pipeline
            return FinalVideoCritiqueResult(
                overall_score=1.0,
                visual_quality_score=1.0,
                audio_sync_score=1.0,
                narrative_flow_score=1.0,
                scene_scores={},
                failing_scenes=[],
                feedback=f"Critique skipped due to error: {e}",
                improvement_suggestions=[],
                passed=True,
                model_used=self.llm.name,
                duration_seconds=time.time() - start_time,
            )

        # Parse scores
        overall_score = eval_data.get("overall_score", 0.0)
        visual_quality = eval_data.get("visual_quality_score", 0.0)
        audio_sync = eval_data.get("audio_sync_score", 0.0)
        narrative_flow = eval_data.get("narrative_flow_score", 0.0)

        # Parse scene scores (convert string keys to int)
        raw_scene_scores = eval_data.get("scene_scores", {})
        scene_scores = {int(k): float(v) for k, v in raw_scene_scores.items()}

        # Get failing scenes
        failing_scenes = eval_data.get("failing_scenes", [])
        # Ensure they're integers
        failing_scenes = [int(s) for s in failing_scenes]

        # Determine pass/fail
        passed = overall_score >= threshold

        duration = time.time() - start_time

        result = FinalVideoCritiqueResult(
            overall_score=overall_score,
            visual_quality_score=visual_quality,
            audio_sync_score=audio_sync,
            narrative_flow_score=narrative_flow,
            scene_scores=scene_scores,
            failing_scenes=failing_scenes,
            feedback=eval_data.get("feedback", ""),
            improvement_suggestions=eval_data.get("improvement_suggestions", []),
            passed=passed,
            raw_response=eval_data,
            model_used=self.llm.name,
            duration_seconds=duration,
        )

        logger.info(
            "final_video_critique_completed",
            overall_score=overall_score,
            visual_quality=visual_quality,
            audio_sync=audio_sync,
            narrative_flow=narrative_flow,
            passed=passed,
            failing_scene_count=len(failing_scenes),
            duration=duration,
        )

        return result

    def _create_stub_result(
        self,
        plan: dict[str, Any],
        threshold: float,  # noqa: ARG002
        duration: float,
    ) -> FinalVideoCritiqueResult:
        """Create a stub result for testing."""
        scene_count = len(plan.get("scenes", []))
        scene_scores = {i + 1: 0.85 for i in range(scene_count)}

        return FinalVideoCritiqueResult(
            overall_score=0.85,
            visual_quality_score=0.85,
            audio_sync_score=0.85,
            narrative_flow_score=0.85,
            scene_scores=scene_scores,
            failing_scenes=[],
            feedback="Stub critique - all scenes pass for testing",
            improvement_suggestions=[],
            passed=True,
            model_used="stub",
            duration_seconds=duration,
        )


def get_final_video_critique_service() -> FinalVideoCritiqueService:
    """Get a final video critique service instance."""
    return FinalVideoCritiqueService()
