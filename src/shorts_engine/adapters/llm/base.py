"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class LLMMessage:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations:
    - OpenAIProvider: Uses OpenAI API (GPT-4, etc.)
    - AnthropicProvider: Uses Anthropic API (Claude)
    - StubLLMProvider: Returns mock data for testing
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion from messages.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            json_mode: If True, request JSON output format

        Returns:
            LLMResponse with generated content
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if provider is operational
        """
        return True
