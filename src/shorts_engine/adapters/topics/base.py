"""Base interface for topic generation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class TopicContext:
    """Context for generating topics.

    Provides the generator with information about the project and
    historical performance to generate relevant, high-performing topics.
    """

    project_id: UUID
    project_name: str
    project_description: str | None = None
    niche: str | None = None

    # Historical performance context
    top_performing_topics: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)  # To avoid duplicates
    top_hook_types: list[str] = field(default_factory=list)
    avg_video_duration_seconds: float = 45.0

    # Optional trend signals
    trending_topics: list[str] = field(default_factory=list)

    # Additional context
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedTopic:
    """A generated topic idea for video creation."""

    topic: str
    hook_suggestion: str | None = None
    estimated_virality_score: float | None = None  # 0-1 confidence score
    reasoning: str | None = None  # Why this topic might perform well
    trend_source: str | None = None  # If based on a trend, what source

    def __str__(self) -> str:
        return self.topic


class TopicProvider(ABC):
    """Abstract base class for topic generation providers.

    Implementations:
    - LLMTopicProvider: Uses LLM to generate topics based on context
    - StubTopicProvider: Returns mock topics for testing
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def generate_topics(
        self,
        context: TopicContext,
        n: int = 5,
        temperature: float = 0.8,
    ) -> list[GeneratedTopic]:
        """Generate topic ideas for video creation.

        Args:
            context: Context about the project and historical performance
            n: Number of topics to generate
            temperature: Creativity level (0-1, higher = more creative)

        Returns:
            List of generated topics with metadata
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if provider is operational
        """
        return True
