"""Learning loop planner for recipe-aware video planning.

Extends the base PlannerService to generate videos that conform to
specific recipe constraints (scene count, hook type, ending type, etc.)
"""

import json
from dataclasses import dataclass

from shorts_engine.adapters.llm.anthropic import AnthropicProvider
from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider
from shorts_engine.adapters.llm.openai import OpenAIProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import settings
from shorts_engine.domain.enums import (
    CaptionDensityBucket,
    EndingType,
    HookType,
    NarrationWPMBucket,
)
from shorts_engine.logging import get_logger
from shorts_engine.presets.styles import StylePreset, get_preset
from shorts_engine.services.learning.context import OptimizationContext
from shorts_engine.services.learning.recipe import Recipe
from shorts_engine.services.planner import ScenePlan, VideoPlan

logger = get_logger(__name__)


@dataclass
class RecipeConstraints:
    """Constraints for video generation from a recipe."""

    scene_count: int
    hook_type: str
    ending_type: str
    narration_wpm_bucket: str
    caption_density_bucket: str

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> "RecipeConstraints":
        """Create constraints from a recipe."""
        return cls(
            scene_count=recipe.scene_count,
            hook_type=recipe.hook_type,
            ending_type=recipe.ending_type,
            narration_wpm_bucket=recipe.narration_wpm_bucket,
            caption_density_bucket=recipe.caption_density_bucket,
        )

    def get_hook_guidance(self) -> str:
        """Get guidance for the hook type."""
        guidance = {
            HookType.QUESTION.value: "Start with a compelling question that creates curiosity",
            HookType.STATEMENT.value: "Open with a bold, attention-grabbing statement or claim",
            HookType.VISUAL.value: "Begin with a striking, eye-catching visual moment",
            HookType.STORY.value: "Start with a mini-story opening that hooks the viewer",
            HookType.CONTRAST.value: "Open with a before/after or comparison setup",
            HookType.MYSTERY.value: "Create a curiosity gap that makes viewers want to watch more",
        }
        return guidance.get(self.hook_type, guidance[HookType.STATEMENT.value])

    def get_ending_guidance(self) -> str:
        """Get guidance for the ending type."""
        guidance = {
            EndingType.CLIFFHANGER.value: "End with a cliffhanger that leaves viewers wanting more, teasing a continuation or unanswered question",
            EndingType.RESOLVE.value: "Provide a satisfying conclusion that wraps up the narrative completely",
            EndingType.CTA.value: "End with a clear call to action (follow, like, comment, etc.)",
            EndingType.LOOP.value: "Create an ending that loops back to the beginning seamlessly",
        }
        return guidance.get(self.ending_type, guidance[EndingType.RESOLVE.value])

    def get_pacing_guidance(self) -> str:
        """Get pacing guidance based on narration and caption density."""
        narration_guidance = {
            NarrationWPMBucket.SLOW.value: "slow, deliberate narration (~100-120 WPM) for emphasis and clarity",
            NarrationWPMBucket.MEDIUM.value: "moderate narration pace (~120-160 WPM) for natural flow",
            NarrationWPMBucket.FAST.value: "quick, energetic narration (~160-200 WPM) for excitement",
        }

        caption_guidance = {
            CaptionDensityBucket.SPARSE.value: "minimal on-screen text, letting visuals speak",
            CaptionDensityBucket.MEDIUM.value: "balanced caption beats at key moments",
            CaptionDensityBucket.DENSE.value: "frequent on-screen text for maximum engagement",
        }

        narration = narration_guidance.get(
            self.narration_wpm_bucket, narration_guidance[NarrationWPMBucket.MEDIUM.value]
        )
        captions = caption_guidance.get(
            self.caption_density_bucket, caption_guidance[CaptionDensityBucket.MEDIUM.value]
        )

        return f"Use {narration} with {captions}"


class LearningLoopPlanner:
    """Planner that generates videos conforming to recipe constraints.

    This extends the base planning approach to:
    1. Enforce exact scene count from recipe
    2. Use specified hook and ending types
    3. Match narration pacing and caption density
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an expert short-form video director and scriptwriter.
Your task is to create compelling video plans for vertical short-form content (TikTok, YouTube Shorts, Instagram Reels).

IMPORTANT CONSTRAINTS - You MUST follow these exactly:
- Create EXACTLY {scene_count} scenes (no more, no less)
- Hook type: {hook_guidance}
- Ending type: {ending_guidance}
- Pacing: {pacing_guidance}

You will receive:
1. A one-paragraph idea describing the video concept
2. A style preset with visual tokens and aesthetic guidelines

You must output a JSON object with this exact structure:
{{
    "title": "Catchy title under 60 characters",
    "description": "SEO-optimized description, 2-3 sentences",
    "scenes": [
        {{
            "scene_number": 1,
            "visual_prompt": "Detailed visual description for AI video generation. Include camera movement, lighting, subject details.",
            "continuity_notes": "Notes for maintaining visual consistency with other scenes",
            "caption_beat": "2-6 word caption/hook for this moment",
            "duration_seconds": 5.0
        }}
        // ... exactly {scene_count} scenes
    ],
    "hook_type": "{hook_type}",
    "ending_type": "{ending_type}"
}}

Guidelines:
- Scene 1 MUST implement the {hook_type} hook style
- The final scene MUST implement the {ending_type} ending style
- Total duration should be approximately 60 seconds
- Visual prompts should be specific and detailed for AI video generation
- Caption beats should be punchy hooks that work as on-screen text
- Incorporate the style tokens naturally into visual prompts
- Build tension/interest with a clear arc from hook to ending"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        default_provider: str = "openai",
    ) -> None:
        """Initialize the planner.

        Args:
            llm_provider: Optional LLM provider. If None, auto-selects based on config.
            default_provider: Default provider type if no API keys configured
        """
        self.llm = llm_provider or self._get_default_provider(default_provider)
        logger.info("learning_loop_planner_initialized", provider=self.llm.name)

    def _get_default_provider(self, default: str) -> LLMProvider:
        """Get the default LLM provider based on available API keys."""
        if settings.openai_api_key:
            return OpenAIProvider(model=settings.openai_model)
        if hasattr(settings, "anthropic_api_key") and settings.anthropic_api_key:
            return AnthropicProvider()

        if default == "anthropic":
            return AnthropicProvider()
        elif default == "openai":
            return OpenAIProvider(model=settings.openai_model)
        else:
            logger.warning("No LLM API keys configured, using stub provider")
            return StubLLMProvider()

    def _build_system_prompt(self, constraints: RecipeConstraints) -> str:
        """Build the system prompt with constraints."""
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            scene_count=constraints.scene_count,
            hook_guidance=constraints.get_hook_guidance(),
            ending_guidance=constraints.get_ending_guidance(),
            pacing_guidance=constraints.get_pacing_guidance(),
            hook_type=constraints.hook_type,
            ending_type=constraints.ending_type,
        )

    def _build_user_prompt(
        self,
        idea: str,
        preset: StylePreset,
        optimization_context: OptimizationContext | None = None,
    ) -> str:
        """Build the user prompt with idea, style context, and optimization learnings."""
        # Base prompt
        prompt = f"""Create a video plan for the following concept:

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

### Target Duration Per Scene:
{preset.default_duration_per_scene} seconds"""

        # Add optimization context if available
        if optimization_context:
            context_text = optimization_context.format_for_prompt()
            if context_text:
                prompt += f"\n\n{context_text}"

        prompt += f"\n\nGenerate a video plan that brings this idea to life in the {preset.display_name} style, following all the constraints provided."

        return prompt

    async def plan(
        self,
        idea: str,
        recipe: Recipe,
        optimization_context: OptimizationContext | None = None,
    ) -> VideoPlan:
        """Generate a video plan that conforms to a recipe.

        Args:
            idea: One-paragraph description of the video concept
            recipe: Recipe defining constraints
            optimization_context: Optional context with learnings from past performance

        Returns:
            VideoPlan with title, description, and scenes

        Raises:
            ValueError: If preset not found or LLM returns invalid response
        """
        # Get the style preset from recipe
        preset = get_preset(recipe.preset)
        if not preset:
            raise ValueError(f"Unknown style preset: {recipe.preset}")

        # Build constraints from recipe
        constraints = RecipeConstraints.from_recipe(recipe)

        logger.info(
            "planning_video_with_recipe",
            idea_length=len(idea),
            recipe_hash=recipe.recipe_hash,
            scene_count=constraints.scene_count,
            hook_type=constraints.hook_type,
            ending_type=constraints.ending_type,
            has_optimization_context=optimization_context is not None,
        )

        # Build messages
        messages = [
            LLMMessage(role="system", content=self._build_system_prompt(constraints)),
            LLMMessage(
                role="user",
                content=self._build_user_prompt(idea, preset, optimization_context),
            ),
        ]

        # Get LLM response
        response = await self.llm.complete(
            messages=messages,
            temperature=0.7,  # Slightly less random for constraint adherence
            max_tokens=4096,
            json_mode=True,
        )

        # Parse JSON response
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error("planner_json_parse_error", error=str(e), content=response.content[:500])
            raise ValueError(f"LLM returned invalid JSON: {e}")

        # Validate response
        if "scenes" not in data or not data["scenes"]:
            raise ValueError("LLM response missing scenes")

        # Validate scene count
        actual_scene_count = len(data["scenes"])
        if actual_scene_count != constraints.scene_count:
            logger.warning(
                "scene_count_mismatch",
                expected=constraints.scene_count,
                actual=actual_scene_count,
            )
            # Adjust scenes to match constraint
            if actual_scene_count < constraints.scene_count:
                # Pad with additional scenes
                last_scene = data["scenes"][-1] if data["scenes"] else {}
                while len(data["scenes"]) < constraints.scene_count:
                    data["scenes"].append(
                        {
                            **last_scene,
                            "scene_number": len(data["scenes"]) + 1,
                            "caption_beat": "...",
                        }
                    )
            else:
                # Trim excess scenes
                data["scenes"] = data["scenes"][: constraints.scene_count]

        # Build ScenePlans
        scenes = []
        for i, scene_data in enumerate(data["scenes"]):
            scenes.append(
                ScenePlan(
                    scene_number=i + 1,
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

        # Build VideoPlan
        plan = VideoPlan(
            title=data.get("title", "Untitled Video"),
            description=data.get("description", ""),
            scenes=scenes,
            style_preset=recipe.preset,
            raw_response={
                **data,
                "recipe_hash": recipe.recipe_hash,
                "hook_type": constraints.hook_type,
                "ending_type": constraints.ending_type,
                "optimization_context_used": optimization_context is not None,
            },
        )

        logger.info(
            "video_planned_with_recipe",
            title=plan.title,
            scene_count=len(plan.scenes),
            total_duration=plan.total_duration,
            recipe_hash=recipe.recipe_hash,
        )

        return plan

    async def health_check(self) -> bool:
        """Check if the planner's LLM provider is healthy."""
        return await self.llm.health_check()
