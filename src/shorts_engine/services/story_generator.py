"""Story generation service for story-first video creation."""

import json
from dataclasses import dataclass

from shorts_engine.adapters.llm.anthropic import AnthropicProvider
from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider
from shorts_engine.adapters.llm.openai import OpenAIProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Story:
    """Generated story for video creation."""

    title: str
    narrative_text: str
    topic: str
    narrative_style: str  # second-person, third-person
    suggested_preset: str
    word_count: int
    estimated_duration_seconds: float


class StoryGenerator:
    """Generates AI-future fantasy stories from topics.

    Creates short, visual stories (100-150 words) that can be turned
    into engaging short-form video content.
    """

    SYSTEM_PROMPT = """You are a thriller writer creating dark, suspenseful near-future AI stories.
Your stories are unsettling, mysterious, and leave viewers thinking. Think Black Mirror meets Twilight Zone.

RULES:
1. Length: 100-150 words (40-60 seconds when narrated at natural pace)
2. ONE PROTAGONIST - an everyday person (use a common first name only, no last names)
3. Build tension and dread - something is WRONG and getting worse
4. The AI or technology should feel threatening, invasive, or too knowing
5. End with a gut-punch twist or unsettling realization - NOT a happy ending
6. Make it VISUAL - describe what we SEE, not just what characters think
7. Present tense, second or third person for immediacy

TONE: Paranoid. Claustrophobic. The technology that was supposed to help is now watching,
controlling, or replacing something human.

AVOID:
- Sappy emotional moments or heartwarming endings
- Happy resolutions where the AI turns out to be helpful
- Multiple characters or jumping between perspectives
- Generic "AI companion becomes my friend" stories
- Anything that feels safe, predictable, or comforting
- First-person narration (use second or third person for distance)

The best stories make viewers uncomfortable. They should want to put down their phone
after watching - but they can't look away.

Return valid JSON with this exact structure:
{
    "title": "Short, evocative title (max 60 characters)",
    "narrative_text": "The full story text (100-150 words)",
    "narrative_style": "second-person|third-person",
    "suggested_preset": "DARK_DYSTOPIAN_ANIME|CINEMATIC_REALISM|SURREAL_DREAMSCAPE"
}

Style preset guidance:
- DARK_DYSTOPIAN_ANIME: Moody, neon-lit, cyberpunk, surveillance aesthetics (PREFERRED)
- CINEMATIC_REALISM: Grounded horror, film-quality, domestic unease
- SURREAL_DREAMSCAPE: Nightmarish, abstract, reality-bending imagery"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the story generator with an LLM provider.

        Args:
            llm_provider: Optional LLM provider. If None, auto-selects based on config.
        """
        self.llm = llm_provider or self._get_default_provider()
        logger.info("story_generator_initialized", provider=self.llm.name)

    def _get_default_provider(self) -> LLMProvider:
        """Get the default LLM provider based on available API keys."""
        provider_name = settings.llm_provider.lower()

        if provider_name == "stub":
            return StubLLMProvider()
        if provider_name == "anthropic" and settings.anthropic_api_key:
            return AnthropicProvider()
        if provider_name == "openai" and settings.openai_api_key:
            return OpenAIProvider()

        # Fallback to checking API keys
        if settings.openai_api_key:
            return OpenAIProvider()
        if settings.anthropic_api_key:
            return AnthropicProvider()

        logger.warning("No LLM API keys configured, using stub provider")
        return StubLLMProvider()

    def _build_user_prompt(self, topic: str) -> str:
        """Build the user prompt for story generation."""
        return f"""Create a dark, suspenseful short story about the following topic:

TOPIC: {topic}

Remember:
- 100-150 words maximum (aim for 120-140 for best pacing)
- ONE protagonist with a common first name
- Build dread - something is WRONG from the start
- Show us what the character SEES, not just thinks
- End with a twist that makes viewers uncomfortable
- The technology should feel invasive or threatening
- NO happy endings, NO AI-turns-out-to-be-good resolutions

Return valid JSON only."""

    async def generate(self, topic: str) -> Story:
        """Generate a story from a topic.

        Args:
            topic: The topic/theme for the story

        Returns:
            Story with title, narrative, and metadata

        Raises:
            ValueError: If LLM returns invalid response
        """
        logger.info(
            "story_generation_started",
            topic=topic[:100],
            llm_provider=self.llm.name,
        )

        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=self._build_user_prompt(topic)),
        ]

        response = await self.llm.complete(
            messages=messages,
            temperature=0.9,  # High creativity for story generation
            max_tokens=2048,
            json_mode=True,
        )

        # Parse JSON response
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(
                "story_json_parse_error",
                error=str(e),
                content=response.content[:500],
            )
            raise ValueError(f"LLM returned invalid JSON: {e}")

        # Validate required fields
        required_fields = ["title", "narrative_text", "narrative_style", "suggested_preset"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"LLM response missing required fields: {missing}")

        # Calculate word count and duration
        narrative_text = data["narrative_text"]
        word_count = len(narrative_text.split())
        estimated_duration = (word_count / settings.story_narration_wpm) * 60

        # Validate word count
        if word_count > settings.story_max_words:
            logger.warning(
                "story_word_count_exceeded",
                word_count=word_count,
                max_words=settings.story_max_words,
            )

        story = Story(
            title=data["title"],
            narrative_text=narrative_text,
            topic=topic,
            narrative_style=data["narrative_style"],
            suggested_preset=data["suggested_preset"],
            word_count=word_count,
            estimated_duration_seconds=round(estimated_duration, 1),
        )

        logger.info(
            "story_generated",
            title=story.title,
            narrative_style=story.narrative_style,
            word_count=story.word_count,
            duration=story.estimated_duration_seconds,
        )

        return story

    async def health_check(self) -> bool:
        """Check if the story generator's LLM provider is healthy."""
        return await self.llm.health_check()
