"""ElevenLabs voiceover provider implementation."""

from typing import Any

import httpx

from shorts_engine.adapters.voiceover.base import (
    VoiceoverProvider,
    VoiceoverRequest,
    VoiceoverResult,
)
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class ElevenLabsProvider(VoiceoverProvider):
    """ElevenLabs API provider for high-quality AI voiceovers.

    ElevenLabs offers natural-sounding AI voices with emotion and style control.
    """

    # Default voice IDs from ElevenLabs
    DEFAULT_VOICES = {
        "narrator": "21m00Tcm4TlvDq8ikWAM",  # Rachel - calm narrator
        "dramatic": "29vD33N1CtxCmqQRPOHJ",  # Drew - dramatic male
        "energetic": "ErXwobaYiN019PkySvjV",  # Antoni - energetic
        "deep": "VR6AewLTigWG4xSOukaG",  # Arnold - deep voice
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "eleven_multilingual_v2",
        base_url: str = "https://api.elevenlabs.io/v1",
    ) -> None:
        self.api_key = api_key or getattr(settings, "elevenlabs_api_key", None)
        self.model_id = model_id
        self.base_url = base_url

        if not self.api_key:
            logger.warning("ElevenLabs API key not configured")

    @property
    def name(self) -> str:
        return "elevenlabs"

    async def generate(self, request: VoiceoverRequest) -> VoiceoverResult:
        """Generate voiceover using ElevenLabs API."""
        if not self.api_key:
            return VoiceoverResult(
                success=False,
                error_message="ElevenLabs API key not configured",
            )

        # Resolve voice ID
        voice_id = request.voice_id or self.DEFAULT_VOICES.get("narrator")
        if request.voice_id in self.DEFAULT_VOICES:
            voice_id = self.DEFAULT_VOICES[request.voice_id]

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Voice settings
        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True,
        }

        # Apply speed adjustment if specified
        if request.speed != 1.0:
            # ElevenLabs doesn't directly support speed, but we can note it
            voice_settings["style"] = min(1.0, request.speed * 0.5)

        payload = {
            "text": request.text,
            "model_id": self.model_id,
            "voice_settings": voice_settings,
        }

        # Add language hint if not English
        if request.language != "en":
            payload["language_code"] = request.language

        logger.info(
            "elevenlabs_generation_started",
            text_length=len(request.text),
            voice_id=voice_id,
            model=self.model_id,
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech/{voice_id}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                audio_data = response.content

            # Estimate duration (rough: ~150 words per minute)
            word_count = len(request.text.split())
            estimated_duration = (word_count / 150) * 60

            logger.info(
                "elevenlabs_generation_completed",
                audio_size=len(audio_data),
                estimated_duration=estimated_duration,
            )

            return VoiceoverResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=estimated_duration,
                metadata={
                    "provider": self.name,
                    "voice_id": voice_id,
                    "model_id": self.model_id,
                    "text_length": len(request.text),
                },
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"ElevenLabs API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"{error_msg} - {error_data.get('detail', {}).get('message', str(error_data))}"
            except Exception:
                pass
            logger.error("elevenlabs_api_error", error=error_msg)
            return VoiceoverResult(success=False, error_message=error_msg)
        except Exception as e:
            logger.error("elevenlabs_generation_error", error=str(e))
            return VoiceoverResult(success=False, error_message=str(e))

    async def list_voices(self) -> list[dict[str, Any]]:
        """List available voices from ElevenLabs."""
        if not self.api_key:
            return []

        headers = {"xi-api-key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/voices",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            return data.get("voices", [])
        except Exception as e:
            logger.error("elevenlabs_list_voices_error", error=str(e))
            return []

    async def health_check(self) -> bool:
        """Check if ElevenLabs API is accessible."""
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/user",
                    headers={"xi-api-key": self.api_key},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error("elevenlabs_health_check_failed", error=str(e))
            return False
