"""Image sequence planner for limited animation style videos.

This service generates a sequence of image prompts (keyframes) from a video plan,
designed for the "limited animation" approach where 8-12 static images with
Ken Burns motion effects create the illusion of animation.
"""

from dataclasses import dataclass, field
from typing import Any

from shorts_engine.adapters.image_gen.base import MotionParams
from shorts_engine.adapters.llm.base import LLMProvider
from shorts_engine.adapters.llm.openai import OpenAIProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import get_settings
from shorts_engine.logging import get_logger
from shorts_engine.presets.styles import StylePreset, get_preset

logger = get_logger(__name__)


@dataclass
class ImageKeyframe:
    """A single keyframe in the image sequence."""

    frame_number: int
    prompt: str
    duration_seconds: float
    motion: MotionParams
    caption_text: str | None = None
    scene_number: int | None = None
    transition_to_next: str = "cut"  # cut, crossfade, fade_to_black


@dataclass
class ImageSequencePlan:
    """Complete plan for an image sequence video."""

    keyframes: list[ImageKeyframe]
    total_duration: float
    style_preset: str
    title: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.keyframes)


class ImageSequencePlanner:
    """Plans image sequences for limited animation videos.

    Takes a video idea and style preset, then generates:
    - 8-12 keyframe prompts optimized for the style
    - Motion parameters for each frame
    - Timing and transitions between frames
    """

    # Target frames based on video duration
    MIN_FRAMES = 6
    MAX_FRAMES = 15
    FRAMES_PER_MINUTE = 10  # ~6 seconds per frame average

    def __init__(self, llm: LLMProvider | None = None) -> None:
        """Initialize the planner.

        Args:
            llm: LLM provider for generating prompts. If None, uses config.
        """
        if llm:
            self.llm = llm
        else:
            settings = get_settings()
            provider = getattr(settings, "llm_provider", "stub").lower()
            if provider == "openai":
                self.llm = OpenAIProvider()
            else:
                self.llm = StubLLMProvider()

    async def plan(
        self,
        idea: str,
        style_preset: str,
        target_duration: float = 60.0,
        optimization_context: str | None = None,
    ) -> ImageSequencePlan:
        """Generate an image sequence plan from an idea.

        Args:
            idea: The video concept/story
            style_preset: Style preset name (e.g., "ATTACK_ON_TITAN")
            target_duration: Target video duration in seconds
            optimization_context: Optional context from learning loop

        Returns:
            ImageSequencePlan with keyframes and timing
        """
        preset = get_preset(style_preset)
        frame_count = self._calculate_frame_count(target_duration)

        logger.info(
            "image_sequence_planning_started",
            idea_length=len(idea),
            style_preset=style_preset,
            target_duration=target_duration,
            frame_count=frame_count,
        )

        # Generate keyframe prompts using LLM
        system_prompt = self._build_system_prompt(preset, frame_count)
        user_prompt = self._build_user_prompt(idea, frame_count, optimization_context)

        response = await self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2000,
            temperature=0.7,
        )

        # Parse the response into keyframes
        keyframes = self._parse_keyframes(
            response.content,
            frame_count,
            target_duration,
            preset,
        )

        # Calculate total duration
        total_duration = sum(kf.duration_seconds for kf in keyframes)

        plan = ImageSequencePlan(
            keyframes=keyframes,
            total_duration=total_duration,
            style_preset=style_preset,
            metadata={
                "idea": idea[:200],
                "llm_model": self.llm.name,
                "target_duration": target_duration,
            },
        )

        logger.info(
            "image_sequence_planning_completed",
            frame_count=len(keyframes),
            total_duration=total_duration,
        )

        return plan

    def _calculate_frame_count(self, target_duration: float) -> int:
        """Calculate optimal number of frames for duration."""
        # Target ~6 seconds per frame for anime pacing
        frames = int(target_duration / 6)
        return max(self.MIN_FRAMES, min(self.MAX_FRAMES, frames))

    def _build_system_prompt(self, preset: StylePreset | None, frame_count: int) -> str:
        """Build the system prompt for the LLM."""
        style_guidance = ""
        if preset:
            style_guidance = f"""
Style: {preset.display_name}
Description: {preset.description}
Visual Keywords: {', '.join(preset.style_tokens[:8])}
Avoid: {', '.join(preset.negative_tokens[:5]) if preset.negative_tokens else 'N/A'}
"""

        return f"""You are a storyboard artist creating keyframes for a short-form animated video.

{style_guidance}

Your task is to create {frame_count} keyframe descriptions that tell a compelling visual story.
Each keyframe should be a detailed image generation prompt.

Guidelines:
1. HOOK - First frame must be visually striking to grab attention in 3 seconds
2. EMOTION - Focus on dramatic expressions and body language
3. COMPOSITION - Use dynamic angles (low angle for power, high angle for vulnerability)
4. CONTINUITY - Maintain consistent character appearances across frames
5. PACING - Vary shot types (close-up, medium, wide) for visual rhythm
6. CLIMAX - Build to a visual peak around frame {frame_count - 2}
7. ENDING - Final frame should be memorable/shareable

For each frame, provide:
- FRAME N: [detailed image prompt, 2-3 sentences]
- CAPTION: [short text overlay, max 10 words]
- SHOT: [close-up / medium / wide / extreme close-up]

Focus on static moments with high emotional impact - anime "freeze frames"."""

    def _build_user_prompt(
        self,
        idea: str,
        frame_count: int,
        optimization_context: str | None = None,
    ) -> str:
        """Build the user prompt for the LLM."""
        prompt = f"""Create {frame_count} keyframes for this video concept:

"{idea}"

"""
        if optimization_context:
            prompt += f"""
Based on past performance data:
{optimization_context}

Apply these learnings to make the visuals more engaging.
"""

        prompt += f"\nGenerate exactly {frame_count} frames in order."
        return prompt

    def _parse_keyframes(
        self,
        llm_response: str,
        expected_count: int,
        target_duration: float,
        preset: StylePreset | None,
    ) -> list[ImageKeyframe]:
        """Parse LLM response into keyframe objects."""
        keyframes: list[ImageKeyframe] = []
        lines = llm_response.strip().split("\n")

        current_frame: dict[str, Any] = {}
        frame_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse frame prompt
            if line.upper().startswith("FRAME"):
                # Save previous frame if exists
                if current_frame.get("prompt"):
                    keyframes.append(self._create_keyframe(current_frame, preset))
                    frame_num += 1

                # Extract prompt after "FRAME N:"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    current_frame = {
                        "frame_number": frame_num,
                        "prompt": parts[1].strip(),
                    }
                else:
                    current_frame = {"frame_number": frame_num, "prompt": ""}

            # Parse caption
            elif line.upper().startswith("CAPTION:"):
                caption = line.split(":", 1)[1].strip() if ":" in line else ""
                current_frame["caption"] = caption

            # Parse shot type
            elif line.upper().startswith("SHOT:"):
                shot = line.split(":", 1)[1].strip().lower() if ":" in line else "medium"
                current_frame["shot_type"] = shot

            # Continuation of prompt
            elif current_frame.get("prompt") is not None and not any(
                line.upper().startswith(x) for x in ["CAPTION", "SHOT", "FRAME"]
            ):
                current_frame["prompt"] += " " + line

        # Don't forget the last frame
        if current_frame.get("prompt"):
            keyframes.append(self._create_keyframe(current_frame, preset))

        # Ensure we have enough frames
        while len(keyframes) < expected_count:
            keyframes.append(
                self._create_keyframe(
                    {
                        "frame_number": len(keyframes),
                        "prompt": "Dramatic scene continuation",
                        "caption": "",
                    },
                    preset,
                )
            )

        # Trim if too many
        keyframes = keyframes[:expected_count]

        # Distribute duration
        duration_per_frame = target_duration / len(keyframes)
        for kf in keyframes:
            kf.duration_seconds = duration_per_frame

        # Adjust first frame (hook) to be slightly shorter
        if keyframes:
            keyframes[0].duration_seconds = min(4.0, duration_per_frame)

        return keyframes

    def _create_keyframe(
        self,
        data: dict[str, Any],
        preset: StylePreset | None,
    ) -> ImageKeyframe:
        """Create a keyframe from parsed data."""
        frame_num = data.get("frame_number", 0)
        prompt = data.get("prompt", "")
        shot_type = data.get("shot_type", "medium")

        # Add style tokens to prompt
        if preset:
            style_suffix = ", ".join(preset.style_tokens[:5])
            prompt = f"{prompt}, {style_suffix}"

        # Get motion parameters based on style and shot type
        motion = self._get_motion_for_shot(shot_type, preset)

        # Determine transition (crossfade for emotional moments)
        transition = "cut"
        if shot_type in ["close-up", "extreme close-up"]:
            transition = "crossfade"

        return ImageKeyframe(
            frame_number=frame_num,
            prompt=prompt,
            duration_seconds=5.0,  # Will be adjusted later
            motion=motion,
            caption_text=data.get("caption"),
            transition_to_next=transition,
        )

    def _get_motion_for_shot(
        self,
        shot_type: str,
        preset: StylePreset | None,
    ) -> MotionParams:
        """Get appropriate motion parameters for a shot type."""
        # Start with preset defaults if available
        base_motion = MotionParams.for_style(preset.name) if preset else MotionParams()

        # Adjust based on shot type
        if shot_type == "close-up":
            # Subtle zoom in for intimacy
            return MotionParams(
                zoom_start=1.0,
                zoom_end=1.08,
                ease=base_motion.ease,
                transition=base_motion.transition,
            )
        elif shot_type == "extreme close-up":
            # Very slow zoom for dramatic effect
            return MotionParams(
                zoom_start=1.0,
                zoom_end=1.05,
                ease="ease-out",
                transition="crossfade",
                transition_duration=0.5,
            )
        elif shot_type == "wide":
            # Slow pan or zoom out for establishing shots
            return MotionParams(
                zoom_start=1.1,
                zoom_end=1.0,
                pan_x_start=-0.05,
                pan_x_end=0.05,
                ease=base_motion.ease,
                transition=base_motion.transition,
            )
        else:  # medium
            return base_motion
