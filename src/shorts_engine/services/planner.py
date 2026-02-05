"""Video planning service using LLM providers."""

import json
from dataclasses import dataclass, field
from typing import Any

from shorts_engine.adapters.llm.anthropic import AnthropicProvider
from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider
from shorts_engine.adapters.llm.openai import OpenAIProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import settings
from shorts_engine.logging import get_logger
from shorts_engine.presets.styles import StylePreset, get_preset

logger = get_logger(__name__)


@dataclass
class ScenePlan:
    """Plan for a single scene in a video."""

    scene_number: int
    visual_prompt: str
    continuity_notes: str
    caption_beat: str
    duration_seconds: float = 5.0


@dataclass
class VideoPlan:
    """Complete plan for a video."""

    title: str
    description: str
    scenes: list[ScenePlan]
    style_preset: str
    total_duration: float = field(init=False)
    raw_response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.total_duration = sum(s.duration_seconds for s in self.scenes)


class PlannerService:
    """Service for generating video plans using LLM.

    The planner takes a one-paragraph idea and a style preset,
    then generates a structured plan with 7-8 scenes.
    """

    SYSTEM_PROMPT = """You are an expert short-form video director and scriptwriter.
Your task is to create compelling video plans for vertical short-form content (TikTok, YouTube Shorts, Instagram Reels).

You will receive:
1. A one-paragraph idea describing the video concept
2. A style preset with visual tokens and aesthetic guidelines
3. A target duration and scene count to match the voiceover length

You must output a JSON object with this exact structure:
{
    "title": "Catchy title under 60 characters",
    "description": "SEO-optimized description, 2-3 sentences",
    "scenes": [
        {
            "scene_number": 1,
            "visual_prompt": "Detailed visual description for AI video generation. Include camera movement, lighting, subject details.",
            "continuity_notes": "Notes for maintaining visual consistency with other scenes (character appearance, color grading, style tokens)",
            "caption_beat": "2-6 word caption/hook for this moment",
            "duration_seconds": 5.0
        }
        // ... scenes as specified in the target scene count
    ]
}

IMPORTANT VISUAL RULES (MUST FOLLOW):
- NEVER include text, letters, numbers, words, titles, or subtitles in visual descriptions
- No signs with readable text, labels, UI elements, or written content in scenes
- Avoid describing screens, monitors, or devices showing text/data
- Focus purely on visual imagery, actions, expressions, and atmosphere
- If a scene involves reading or writing, describe the ACTION without specifying visible text

CHARACTER CONSISTENCY RULES:
- In scene 1, define ONE detailed character description including: physical appearance, clothing, distinctive features, hair style/color
- Copy the EXACT same character description word-for-word to every subsequent scene's visual_prompt
- Never use synonyms, abbreviations, or variations for character attributes
- Maintain consistent: lighting direction, color grading, camera style, environment type
- If introducing new characters, give them equally detailed and consistent descriptions

PROMPT STRUCTURE FOR EACH SCENE:
1. Camera/shot description (angle, movement, framing)
2. Character description (IDENTICAL across all scenes - copy-paste from scene 1)
3. Setting/environment (consistent lighting and atmosphere)
4. Specific action or motion for this scene only

Guidelines:
- Create EXACTLY the number of scenes specified in the target scene count
- Each scene should match the specified duration per scene
- Visual prompts should be specific and detailed for AI video generation
- Continuity notes should ensure consistent character/style across scenes
- Caption beats should be punchy hooks that work as on-screen text
- Incorporate the style tokens naturally into visual prompts
- Build tension/interest with a clear arc: hook -> develop -> climax -> resolve"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        default_provider: str = "openai",
    ) -> None:
        """Initialize the planner with an LLM provider.

        Args:
            llm_provider: Optional LLM provider. If None, auto-selects based on config.
            default_provider: Default provider type if no API keys configured ("openai", "anthropic", "stub")
        """
        self.llm = llm_provider or self._get_default_provider(default_provider)
        logger.info("planner_initialized", provider=self.llm.name)

    def _get_default_provider(self, default: str) -> LLMProvider:
        """Get the default LLM provider based on available API keys."""
        # Check for API keys in order of preference
        if settings.openai_api_key:
            return OpenAIProvider(model=settings.openai_model)
        if hasattr(settings, "anthropic_api_key") and settings.anthropic_api_key:
            return AnthropicProvider()

        # Fall back to specified default or stub
        if default == "anthropic":
            return AnthropicProvider()
        elif default == "openai":
            return OpenAIProvider(model=settings.openai_model)
        else:
            logger.warning("No LLM API keys configured, using stub provider")
            return StubLLMProvider()

    def _build_user_prompt(
        self,
        idea: str,
        preset: StylePreset,
        optimization_context: str | None = None,
        story_context: dict[str, str] | None = None,
        target_duration_seconds: float | None = None,
    ) -> str:
        """Build the user prompt with idea and style context.

        Args:
            idea: The video idea (may be full story narrative)
            preset: Style preset for visual guidance
            optimization_context: Optional learnings from past performance
            story_context: Optional dict with 'narrative_style' and 'topic' for richer context
            target_duration_seconds: Optional target total video duration (from voiceover)
        """
        optimization_section = ""
        if optimization_context:
            optimization_section = f"""
{optimization_context}

Use these learnings to inform your creative choices. Patterns that work should be incorporated; patterns to avoid should be steered away from.

"""

        story_section = ""
        if story_context:
            narrative_style = story_context.get("narrative_style", "")
            topic = story_context.get("topic", "")
            if narrative_style or topic:
                story_section = """
## Story Context
This video is based on a pre-written story. Preserve the narrative arc and emotional beats.
"""
                if topic:
                    story_section += f"Original Topic: {topic}\n"
                if narrative_style:
                    story_section += f"Narrative Style: {narrative_style} (maintain this perspective in captions)\n"

        # Calculate scene count and duration based on target
        duration_per_scene = preset.default_duration_per_scene
        if target_duration_seconds:
            scene_count = max(8, int(target_duration_seconds / duration_per_scene))
            # Recalculate duration per scene to match target exactly
            duration_per_scene = round(target_duration_seconds / scene_count, 1)
            duration_section = f"""### Target Duration Requirements:
- Total video duration: {target_duration_seconds} seconds (MUST MATCH)
- Number of scenes: {scene_count} (create EXACTLY this many)
- Duration per scene: {duration_per_scene} seconds each
"""
        else:
            scene_count = 8
            duration_section = f"""### Target Duration Per Scene:
{duration_per_scene} seconds
"""

        return f"""Create a video plan for the following concept:
{optimization_section}
{story_section}
## Video Idea
{idea}

## Style Preset: {preset.display_name}
{preset.description}

### Visual Style Tokens (incorporate into every scene):
{preset.format_style_prompt()}

### Continuity Tokens (use in continuity_notes):
{preset.format_continuity_prompt()}

### Negative Prompts (avoid these):
{preset.format_negative_prompt()}

### Camera Style:
{preset.camera_style}

{duration_section}

Generate a compelling {scene_count}-scene video plan that brings this idea to life in the {preset.display_name} style."""

    async def plan(
        self,
        idea: str,
        style_preset_name: str,
        optimization_context: str | None = None,
        story_context: dict[str, str] | None = None,
        target_duration_seconds: float | None = None,
    ) -> VideoPlan:
        """Generate a video plan from an idea and style preset.

        Args:
            idea: One-paragraph description of the video concept (or full story narrative)
            style_preset_name: Name of the style preset to use
            optimization_context: Optional formatted string with learnings from past performance
            story_context: Optional dict with 'narrative_style' and 'topic' from Story
            target_duration_seconds: Optional target total video duration (from voiceover)
                If provided, scene count and durations will be calculated to match

        Returns:
            VideoPlan with title, description, and scenes

        Raises:
            ValueError: If preset not found or LLM returns invalid response
        """
        # Get the style preset
        preset = get_preset(style_preset_name)
        if not preset:
            available = ", ".join(
                p.name
                for name in ["DARK_DYSTOPIAN_ANIME", "VIBRANT_MOTION_GRAPHICS", "CINEMATIC_REALISM"]
                if (p := get_preset(name))
            )
            raise ValueError(f"Unknown style preset: {style_preset_name}. Available: {available}")

        logger.info(
            "planning_video",
            idea_length=len(idea),
            style_preset=preset.name,
            llm_provider=self.llm.name,
            has_optimization_context=optimization_context is not None,
            has_story_context=story_context is not None,
            target_duration=target_duration_seconds,
        )

        # Build messages
        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=self._build_user_prompt(
                    idea, preset, optimization_context, story_context, target_duration_seconds
                ),
            ),
        ]

        # Get LLM response
        response = await self.llm.complete(
            messages=messages,
            temperature=0.8,  # Some creativity
            max_tokens=4096,
            json_mode=True,
        )

        # Parse JSON response
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error("planner_json_parse_error", error=str(e), content=response.content[:500])
            raise ValueError(f"LLM returned invalid JSON: {e}")

        # Validate and build VideoPlan
        if "scenes" not in data or not data["scenes"]:
            raise ValueError("LLM response missing scenes")

        scenes = []
        for i, scene_data in enumerate(data["scenes"]):
            scenes.append(
                ScenePlan(
                    scene_number=scene_data.get("scene_number", i + 1),
                    visual_prompt=scene_data.get("visual_prompt", ""),
                    continuity_notes=scene_data.get(
                        "continuity_notes", preset.format_continuity_prompt()
                    ),
                    caption_beat=scene_data.get("caption_beat", ""),
                    duration_seconds=float(
                        scene_data.get("duration_seconds", preset.default_duration_per_scene)
                    ),
                )
            )

        # Adjust scene durations to match target duration if specified
        if target_duration_seconds and scenes:
            duration_per_scene = round(target_duration_seconds / len(scenes), 2)
            for scene in scenes:
                scene.duration_seconds = duration_per_scene
            logger.info(
                "adjusted_scene_durations",
                target_duration=target_duration_seconds,
                scene_count=len(scenes),
                duration_per_scene=duration_per_scene,
            )

        plan = VideoPlan(
            title=data.get("title", "Untitled Video"),
            description=data.get("description", ""),
            scenes=scenes,
            style_preset=preset.name,
            raw_response=data,
        )

        logger.info(
            "video_planned",
            title=plan.title,
            scene_count=len(plan.scenes),
            total_duration=plan.total_duration,
        )

        return plan

    async def health_check(self) -> bool:
        """Check if the planner's LLM provider is healthy."""
        return await self.llm.health_check()
