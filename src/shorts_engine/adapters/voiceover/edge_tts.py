"""Microsoft Edge TTS voiceover provider (free fallback)."""

from typing import Any

from shorts_engine.adapters.voiceover.base import (
    VoiceoverProvider,
    VoiceoverRequest,
    VoiceoverResult,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class EdgeTTSProvider(VoiceoverProvider):
    """Microsoft Edge TTS provider - free text-to-speech fallback.

    Uses the edge-tts library to access Microsoft's free TTS service.
    Good quality for a free option, supports many languages and voices.
    """

    # Popular English voices (Multilingual Neural voices for higher quality)
    DEFAULT_VOICES = {
        "narrator": "en-US-AndrewMultilingualNeural",  # Deep, natural male narrator
        "narrator_female": "en-US-AvaMultilingualNeural",  # Natural female narrator
        "dramatic": "en-US-GuyNeural",  # Deep dramatic male
        "thriller": "en-US-AndrewMultilingualNeural",  # Deep voice, good for dark content
        "energetic": "en-US-AriaNeural",  # Energetic female
        "mysterious": "en-GB-RyanNeural",  # British accent for mystery
        "calm_tension": "en-US-BrandonMultilingualNeural",  # Calm but intense
    }

    def __init__(self, default_voice: str = "en-US-AndrewMultilingualNeural") -> None:
        self.default_voice = default_voice

    @property
    def name(self) -> str:
        return "edge_tts"

    async def generate(self, request: VoiceoverRequest) -> VoiceoverResult:
        """Generate voiceover using Microsoft Edge TTS."""
        try:
            # Import edge_tts dynamically (optional dependency)
            import edge_tts  # type: ignore[import-not-found]
        except ImportError:
            return VoiceoverResult(
                success=False,
                error_message="edge-tts package not installed. Run: pip install edge-tts",
            )

        # Resolve voice
        voice = request.voice_id or self.default_voice
        if request.voice_id in self.DEFAULT_VOICES:
            voice = self.DEFAULT_VOICES[request.voice_id]

        # Map language to voice if needed
        if request.language != "en" and not request.voice_id:
            voice = self._get_voice_for_language(request.language)

        logger.info(
            "edge_tts_generation_started",
            text_length=len(request.text),
            voice=voice,
        )

        try:
            # Stream audio + word boundaries so we can generate subtitle timing
            # from the voice track itself (audio-first captions).
            communicate = edge_tts.Communicate(
                text=request.text,
                voice=voice,
                rate=self._format_rate(request.speed),
                pitch=self._format_pitch(request.pitch),
                boundary="WordBoundary",
            )

            audio_chunks: list[bytes] = []
            word_boundaries: list[dict[str, Any]] = []
            sentence_boundaries: list[dict[str, Any]] = []

            async for chunk in communicate.stream():
                chunk_type = chunk.get("type")

                if chunk_type == "audio":
                    audio_data = chunk.get("data")
                    if isinstance(audio_data, bytes):
                        audio_chunks.append(audio_data)
                    continue

                # Edge-TTS offsets are in 100ns units.
                if chunk_type in ("WordBoundary", "SentenceBoundary"):
                    text = str(chunk.get("text") or "").strip()
                    offset = float(chunk.get("offset") or 0.0) / 10_000_000.0
                    duration = float(chunk.get("duration") or 0.0) / 10_000_000.0
                    end = max(offset + duration, offset + 0.04)

                    payload = {
                        "text": text,
                        "start_seconds": offset,
                        "end_seconds": end,
                    }

                    if chunk_type == "WordBoundary" and text:
                        word_boundaries.append(payload)
                    elif chunk_type == "SentenceBoundary" and text:
                        sentence_boundaries.append(payload)

            audio_data = b"".join(audio_chunks)
            if not audio_data:
                raise RuntimeError("edge-tts stream returned no audio chunks")

            # Prefer exact duration from final boundary, fallback to words-per-minute estimate.
            estimated_duration = 0.0
            if word_boundaries:
                estimated_duration = float(word_boundaries[-1]["end_seconds"])
            elif sentence_boundaries:
                estimated_duration = float(sentence_boundaries[-1]["end_seconds"])
            else:
                word_count = len(request.text.split())
                estimated_duration = (word_count / 150) * 60 / request.speed

            logger.info(
                "edge_tts_generation_completed",
                audio_size=len(audio_data),
                estimated_duration=estimated_duration,
                word_boundaries=len(word_boundaries),
            )

            return VoiceoverResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=estimated_duration,
                metadata={
                    "provider": self.name,
                    "voice": voice,
                    "text_length": len(request.text),
                    "word_boundaries": word_boundaries,
                    "sentence_boundaries": sentence_boundaries,
                },
            )

        except Exception as e:
            logger.error("edge_tts_generation_error", error=str(e))
            return VoiceoverResult(success=False, error_message=str(e))

    async def list_voices(self) -> list[dict[str, Any]]:
        """List available voices from Edge TTS."""
        try:
            import edge_tts

            voices = await edge_tts.list_voices()
            return [
                {
                    "voice_id": v["ShortName"],
                    "name": v["FriendlyName"],
                    "language": v["Locale"],
                    "gender": v["Gender"],
                }
                for v in voices
            ]
        except ImportError:
            return []
        except Exception as e:
            logger.error("edge_tts_list_voices_error", error=str(e))
            return []

    async def health_check(self) -> bool:
        """Check if Edge TTS is available."""
        try:
            import edge_tts

            # Try to list voices as a health check
            voices = await edge_tts.list_voices()
            return len(voices) > 0
        except Exception:
            return False

    def _format_rate(self, speed: float) -> str:
        """Format speed as Edge TTS rate string."""
        # Edge TTS uses percentage: +50% = 1.5x speed
        percentage = int((speed - 1.0) * 100)
        if percentage >= 0:
            return f"+{percentage}%"
        return f"{percentage}%"

    def _format_pitch(self, pitch: float) -> str:
        """Format pitch as Edge TTS pitch string."""
        # Edge TTS uses Hz offset: +50Hz
        hz_offset = int((pitch - 1.0) * 50)
        if hz_offset >= 0:
            return f"+{hz_offset}Hz"
        return f"{hz_offset}Hz"

    def _get_voice_for_language(self, language: str) -> str:
        """Get a default voice for a language code."""
        language_voices = {
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural",
            "it": "it-IT-DiegoNeural",
            "pt": "pt-BR-AntonioNeural",
            "ja": "ja-JP-KeitaNeural",
            "ko": "ko-KR-InJoonNeural",
            "zh": "zh-CN-YunxiNeural",
        }
        return language_voices.get(language, self.default_voice)
