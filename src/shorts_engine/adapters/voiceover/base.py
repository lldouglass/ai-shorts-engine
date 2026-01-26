"""Base interface for voiceover generation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VoiceoverRequest:
    """Request for voiceover generation."""

    text: str
    voice_id: str | None = None  # Provider-specific voice identifier
    language: str = "en"
    speed: float = 1.0  # Speech speed multiplier
    pitch: float = 1.0  # Pitch adjustment
    output_format: str = "mp3"
    options: dict[str, Any] | None = None


@dataclass
class VoiceoverResult:
    """Result from voiceover generation."""

    success: bool
    audio_data: bytes | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceoverProvider(ABC):
    """Abstract base class for voiceover generation providers.

    Implementations:
    - ElevenLabsProvider: High-quality AI voices via ElevenLabs API
    - EdgeTTSProvider: Free Microsoft Edge TTS (fallback)
    - StubVoiceoverProvider: Returns mock data for testing
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def generate(self, request: VoiceoverRequest) -> VoiceoverResult:
        """Generate voiceover audio from text.

        Args:
            request: Voiceover request with text and voice settings

        Returns:
            VoiceoverResult with audio data or error information
        """
        ...

    @abstractmethod
    async def list_voices(self) -> list[dict[str, Any]]:
        """List available voices.

        Returns:
            List of voice information dictionaries
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is available and healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        return True
