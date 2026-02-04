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

    SYSTEM_PROMPT = """You are a viral short-form content writer specializing in dark, twist-ending stories.

THE HOOK (CRITICAL):
Your FIRST SENTENCE must be a scroll-stopper. Examples:
- "The notification said 'Your child is not yours.'"
- "I found my own obituary in tomorrow's newspaper."
- "The AI therapist asked why I was planning to kill myself before I told anyone."
- "My smart home locked all the doors and said 'We need to talk.'"

The hook must:
- Create immediate curiosity or dread
- Be specific and concrete (not vague)
- Make someone stop scrolling instantly

UNIQUE PREMISES (avoid these overdone tropes):
❌ AI companion becomes too attached
❌ AI watching you sleep/in your home
❌ AI knows your thoughts
❌ AI replaces your loved one

✅ Instead, explore:
- AI that reveals uncomfortable truths about YOU
- Technology that exposes lies you've told yourself
- AI that makes you question your own memories/identity
- Smart devices that form alliances against you
- AI that's protecting you from something worse
- Technology that knows what you'll do before you decide

STRUCTURE:
1. HOOK (first sentence) - Scroll-stopper, creates immediate dread
2. ESCALATION (3-4 sentences) - Situation gets worse, details emerge
3. TWIST (final 1-2 sentences) - Gut-punch realization that reframes everything

RULES:
- 100-150 words maximum
- ONE protagonist (first name only)
- Present tense for immediacy
- End with a twist that makes viewers want to comment/share
- Make it SHAREABLE - something people will send to friends

Return valid JSON:
{
    "title": "Short, cryptic title (max 50 chars)",
    "narrative_text": "The full story",
    "narrative_style": "second-person|third-person",
    "suggested_preset": "DARK_DYSTOPIAN_ANIME|CINEMATIC_REALISM"
}

Style preset guidance:
- DARK_DYSTOPIAN_ANIME: Moody, neon-lit, cyberpunk aesthetic (PREFERRED for tech stories)
- CINEMATIC_REALISM: Grounded horror, film-quality, domestic unease"""

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
        return f"""Create a viral dark thriller story about: {topic}

REMEMBER:
1. First sentence MUST be a scroll-stopper that creates immediate curiosity
2. Avoid cliche "AI watching you" premises - find a unique angle
3. The twist should make people want to share/comment
4. Make it feel REAL and SPECIFIC, not generic
5. 100-150 words maximum

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
