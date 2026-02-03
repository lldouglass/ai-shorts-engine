"""Tests for the StoryGenerator service."""

import pytest

from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.services.story_generator import Story, StoryGenerator


@pytest.fixture
def stub_llm():
    """Get a stub LLM provider."""
    return StubLLMProvider()


@pytest.fixture
def story_generator(stub_llm):
    """Get a StoryGenerator with stub LLM."""
    return StoryGenerator(llm_provider=stub_llm)


class TestStory:
    """Tests for the Story dataclass."""

    def test_story_creation(self):
        """Test creating a Story instance."""
        story = Story(
            title="Test Title",
            narrative_text="This is a test story with some words.",
            topic="test topic",
            narrative_style="first-person",
            suggested_preset="DARK_DYSTOPIAN_ANIME",
            word_count=8,
            estimated_duration_seconds=3.2,
        )

        assert story.title == "Test Title"
        assert story.narrative_style == "first-person"
        assert story.word_count == 8
        assert story.estimated_duration_seconds == 3.2


class TestStoryGenerator:
    """Tests for the StoryGenerator service."""

    @pytest.mark.asyncio
    async def test_generate_returns_story(self, story_generator):
        """Test that generate returns a valid Story object."""
        story = await story_generator.generate("AI companions in 2050")

        assert isinstance(story, Story)
        assert story.title
        assert story.narrative_text
        assert story.narrative_style in ("first-person", "third-person", "documentary")
        assert story.suggested_preset

    @pytest.mark.asyncio
    async def test_generate_calculates_word_count(self, story_generator):
        """Test that word count is correctly calculated."""
        story = await story_generator.generate("AI companions in 2050")

        # Verify word count matches actual words
        actual_words = len(story.narrative_text.split())
        assert story.word_count == actual_words

    @pytest.mark.asyncio
    async def test_generate_calculates_duration(self, story_generator):
        """Test that duration is estimated based on word count and WPM."""
        story = await story_generator.generate("AI companions in 2050")

        # Duration should be (word_count / wpm) * 60 seconds
        # Default WPM is 150
        expected_duration = (story.word_count / 150) * 60
        assert abs(story.estimated_duration_seconds - expected_duration) < 0.1

    @pytest.mark.asyncio
    async def test_generate_preserves_topic(self, story_generator):
        """Test that the original topic is preserved in the story."""
        topic = "Robot artists creating masterpieces"
        story = await story_generator.generate(topic)

        assert story.topic == topic

    @pytest.mark.asyncio
    async def test_health_check(self, story_generator):
        """Test that health check works with stub provider."""
        is_healthy = await story_generator.health_check()
        assert is_healthy is True


class TestStoryGeneratorProviderSelection:
    """Tests for LLM provider selection in StoryGenerator."""

    def test_uses_provided_provider(self, stub_llm):
        """Test that an explicitly provided provider is used."""
        generator = StoryGenerator(llm_provider=stub_llm)
        assert generator.llm is stub_llm

    def test_default_provider_is_stub_when_no_keys(self, monkeypatch):
        """Test that stub provider is used when no API keys are configured."""
        # Clear any API keys
        monkeypatch.setattr("shorts_engine.config.settings.openai_api_key", None)
        monkeypatch.setattr("shorts_engine.config.settings.anthropic_api_key", None)
        monkeypatch.setattr("shorts_engine.config.settings.llm_provider", "openai")

        generator = StoryGenerator()
        assert generator.llm.name == "stub"


class TestStubLLMStoryGeneration:
    """Tests for the stub LLM's story generation responses."""

    @pytest.mark.asyncio
    async def test_stub_returns_valid_story_json(self):
        """Test that stub LLM returns valid story JSON for story requests."""
        from shorts_engine.adapters.llm.base import LLMMessage

        llm = StubLLMProvider()

        # System prompt with story generation markers
        messages = [
            LLMMessage(
                role="system",
                content="You are a creative writer...",
            ),
            LLMMessage(
                role="user",
                content="Create a short story about the following topic:\n\n"
                "TOPIC: AI companions in 2050\n\n"
                "Remember:\n- 100-150 words maximum",
            ),
        ]

        response = await llm.complete(messages, json_mode=True)

        import json

        data = json.loads(response.content)

        assert "title" in data
        assert "narrative_text" in data
        assert "narrative_style" in data
        assert "suggested_preset" in data

    @pytest.mark.asyncio
    async def test_stub_returns_video_plan_for_non_story(self):
        """Test that stub LLM returns video plan JSON for non-story requests."""
        from shorts_engine.adapters.llm.base import LLMMessage

        llm = StubLLMProvider()

        messages = [
            LLMMessage(
                role="system",
                content="You are an expert video director...",
            ),
            LLMMessage(
                role="user",
                content="Create a video plan for: Epic samurai battle",
            ),
        ]

        response = await llm.complete(messages, json_mode=True)

        import json

        data = json.loads(response.content)

        # Should be a video plan, not a story
        assert "scenes" in data
        assert "title" in data
        assert "narrative_text" not in data
