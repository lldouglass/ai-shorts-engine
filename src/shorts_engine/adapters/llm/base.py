"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class VisionMessage:
    """A message that can include images for vision-capable models."""

    role: str  # "system", "user", "assistant"
    text: str
    image_urls: list[str] = field(default_factory=list)  # URLs or base64 data URIs


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

    async def complete_with_vision(
        self,
        messages: list[VisionMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion from messages that may include images.

        Args:
            messages: List of vision messages with optional images
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            json_mode: If True, request JSON output format

        Returns:
            LLMResponse with generated content

        Raises:
            NotImplementedError: If the provider doesn't support vision
        """
        raise NotImplementedError(f"{self.name} does not support vision")

    @property
    def supports_vision(self) -> bool:
        """Check if this provider supports vision (image) inputs.

        Returns:
            True if the provider can process images
        """
        return False

    @property
    def supports_video(self) -> bool:
        """Check if this provider supports native video input.

        Returns:
            True if the provider can process video files natively
        """
        return False

    async def complete_with_video(
        self,
        video_path: Path,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Analyze a video file natively.

        Args:
            video_path: Path to the video file
            prompt: The prompt to use for analysis
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            json_mode: If True, request JSON output format

        Returns:
            LLMResponse with generated content

        Raises:
            NotImplementedError: If the provider doesn't support video
        """
        raise NotImplementedError(f"{self.name} does not support native video input")

    async def health_check(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if provider is operational
        """
        return True
