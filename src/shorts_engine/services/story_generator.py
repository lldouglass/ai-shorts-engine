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
    narrative_style: str  # first-person, third-person, documentary
    suggested_preset: str
    word_count: int
    estimated_duration_seconds: float


class StoryGenerator:
    """Generates AI-future fantasy stories from topics.

    Creates short, visual stories (100-150 words) that can be turned
    into engaging short-form video content.
    """

    SYSTEM_PROMPT = """You are a creative writer specializing in near-future AI fiction.
Your stories explore life after AI advancement from the perspective of people living in that world.

Requirements for every story:
1. Length: 100-150 words (under 60 seconds when narrated)
2. Visual storytelling: Describe scenes that can be animated/filmed
3. Include 2-3 distinct "moments" that could become video scenes
4. Start with an emotional hook in the first sentence
5. End with a thought-provoking beat (not necessarily a resolution)
6. Feel authentic - like a real person's experience in an AI world

Choose the narrative style that best fits the topic:
- first-person: Personal, intimate stories ("I woke up to...")
- third-person: Character-focused narratives ("Sarah discovered...")
- documentary: World-building, explanatory ("In 2050, humanity...")

Return valid JSON with this exact structure:
{
    "title": "Short, evocative title (max 60 characters)",
    "narrative_text": "The full story text (100-150 words)",
    "narrative_style": "first-person|third-person|documentary",
    "suggested_preset": "DARK_DYSTOPIAN_ANIME|CINEMATIC_REALISM|VIBRANT_MOTION_GRAPHICS|SURREAL_DREAMSCAPE"
}

Style preset guidance:
- DARK_DYSTOPIAN_ANIME: Moody, neon-lit, cyberpunk aesthetics
- CINEMATIC_REALISM: Grounded, film-quality, emotional depth
- VIBRANT_MOTION_GRAPHICS: Bold colors, dynamic, energetic
- SURREAL_DREAMSCAPE: Abstract, ethereal, dreamlike imagery"""

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
        return f"""Create a short story about the following topic:

TOPIC: {topic}

Remember:
- 100-150 words maximum
- Visual, animatable scenes
- Clear emotional hook at the start
- Thought-provoking ending
- Authentic human experience in an AI world

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
