"""Voiceover generation adapters."""

from shorts_engine.adapters.voiceover.base import (
    VoiceoverProvider,
    VoiceoverRequest,
    VoiceoverResult,
)
from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider
from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider
from shorts_engine.adapters.voiceover.stub import StubVoiceoverProvider

__all__ = [
    "VoiceoverProvider",
    "VoiceoverRequest",
    "VoiceoverResult",
    "ElevenLabsProvider",
    "EdgeTTSProvider",
    "StubVoiceoverProvider",
]
