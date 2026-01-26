"""Stub voiceover provider for testing."""

import asyncio
from typing import Any

from shorts_engine.adapters.voiceover.base import (
    VoiceoverProvider,
    VoiceoverRequest,
    VoiceoverResult,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class StubVoiceoverProvider(VoiceoverProvider):
    """Stub provider that simulates voiceover generation without external calls."""

    @property
    def name(self) -> str:
        return "stub"

    async def generate(self, request: VoiceoverRequest) -> VoiceoverResult:
        """Simulate voiceover generation with a delay."""
        logger.info(
            "stub_voiceover_generation_started",
            text_length=len(request.text),
            voice=request.voice_id,
        )

        # Simulate processing time
        await asyncio.sleep(0.2)

        # Create fake audio data (just a marker)
        fake_audio = b"STUB_AUDIO_DATA_" + request.text.encode()[:100]

        # Estimate duration (rough: ~150 words per minute)
        word_count = len(request.text.split())
        estimated_duration = (word_count / 150) * 60

        logger.info(
            "stub_voiceover_generation_completed",
            audio_size=len(fake_audio),
            duration=estimated_duration,
        )

        return VoiceoverResult(
            success=True,
            audio_data=fake_audio,
            duration_seconds=estimated_duration,
            metadata={
                "provider": self.name,
                "voice_id": request.voice_id or "default",
                "text_length": len(request.text),
            },
        )

    async def list_voices(self) -> list[dict[str, Any]]:
        """Return stub voice list."""
        return [
            {
                "voice_id": "stub_narrator",
                "name": "Stub Narrator",
                "language": "en-US",
                "gender": "neutral",
            },
            {
                "voice_id": "stub_dramatic",
                "name": "Stub Dramatic",
                "language": "en-US",
                "gender": "male",
            },
        ]

    async def health_check(self) -> bool:
        """Stub provider is always healthy."""
        return True
