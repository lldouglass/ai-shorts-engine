"""Unit tests for voiceover providers."""

import pytest

from shorts_engine.adapters.voiceover.base import VoiceoverRequest, VoiceoverResult
from shorts_engine.adapters.voiceover.stub import StubVoiceoverProvider


class TestVoiceoverRequest:
    """Tests for VoiceoverRequest dataclass."""

    def test_request_defaults(self):
        """Test default values for voiceover request."""
        request = VoiceoverRequest(text="Hello world")

        assert request.text == "Hello world"
        assert request.voice_id is None
        assert request.language == "en"
        assert request.speed == 1.0
        assert request.pitch == 1.0
        assert request.output_format == "mp3"
        assert request.options is None

    def test_request_with_all_options(self):
        """Test request with all options specified."""
        request = VoiceoverRequest(
            text="Test narration",
            voice_id="narrator",
            language="es",
            speed=1.2,
            pitch=0.9,
            output_format="wav",
            options={"style": "dramatic"},
        )

        assert request.text == "Test narration"
        assert request.voice_id == "narrator"
        assert request.language == "es"
        assert request.speed == 1.2
        assert request.pitch == 0.9
        assert request.output_format == "wav"
        assert request.options == {"style": "dramatic"}


class TestVoiceoverResult:
    """Tests for VoiceoverResult dataclass."""

    def test_success_result(self):
        """Test successful result creation."""
        result = VoiceoverResult(
            success=True,
            audio_data=b"fake_audio_data",
            duration_seconds=10.5,
            metadata={"provider": "test"},
        )

        assert result.success is True
        assert result.audio_data == b"fake_audio_data"
        assert result.duration_seconds == 10.5
        assert result.error_message is None
        assert result.metadata == {"provider": "test"}

    def test_failure_result(self):
        """Test failed result creation."""
        result = VoiceoverResult(
            success=False,
            error_message="API key invalid",
        )

        assert result.success is False
        assert result.audio_data is None
        assert result.error_message == "API key invalid"


class TestStubVoiceoverProvider:
    """Tests for StubVoiceoverProvider."""

    @pytest.fixture
    def provider(self):
        """Create a stub provider instance."""
        return StubVoiceoverProvider()

    def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.name == "stub"

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful voiceover generation."""
        request = VoiceoverRequest(
            text="This is a test narration for the video.",
            voice_id="narrator",
        )

        result = await provider.generate(request)

        assert result.success is True
        assert result.audio_data is not None
        assert len(result.audio_data) > 0
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0
        assert result.metadata["provider"] == "stub"

    @pytest.mark.asyncio
    async def test_generate_estimates_duration(self, provider):
        """Test that duration is estimated from text length."""
        # Short text
        short_request = VoiceoverRequest(text="Hello")
        short_result = await provider.generate(short_request)

        # Long text
        long_request = VoiceoverRequest(text="This is a much longer narration " * 10)
        long_result = await provider.generate(long_request)

        # Longer text should have longer duration
        assert long_result.duration_seconds > short_result.duration_seconds

    @pytest.mark.asyncio
    async def test_list_voices(self, provider):
        """Test listing available voices."""
        voices = await provider.list_voices()

        assert isinstance(voices, list)
        assert len(voices) >= 1

        # Check voice structure
        voice = voices[0]
        assert "voice_id" in voice
        assert "name" in voice

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        """Test health check returns True for stub."""
        is_healthy = await provider.health_check()
        assert is_healthy is True


class TestEdgeTTSProvider:
    """Tests for EdgeTTSProvider (requires edge-tts package)."""

    def test_provider_name(self):
        """Test provider name property."""
        from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider

        provider = EdgeTTSProvider()
        assert provider.name == "edge_tts"

    def test_default_voices(self):
        """Test default voice mappings."""
        from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider

        provider = EdgeTTSProvider()
        assert "narrator" in provider.DEFAULT_VOICES
        assert "narrator_female" in provider.DEFAULT_VOICES

    def test_format_rate(self):
        """Test speed formatting."""
        from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider

        provider = EdgeTTSProvider()

        # Normal speed
        assert provider._format_rate(1.0) == "+0%"

        # Faster
        assert provider._format_rate(1.5) == "+50%"

        # Slower (int rounding may vary, check range)
        slower_rate = provider._format_rate(0.8)
        assert slower_rate in ["-19%", "-20%"]

    def test_format_pitch(self):
        """Test pitch formatting."""
        from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider

        provider = EdgeTTSProvider()

        # Normal pitch
        assert provider._format_pitch(1.0) == "+0Hz"

        # Higher
        assert provider._format_pitch(1.5) == "+25Hz"

        # Lower
        assert provider._format_pitch(0.5) == "-25Hz"

    def test_get_voice_for_language(self):
        """Test language-to-voice mapping."""
        from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider

        provider = EdgeTTSProvider()

        # Spanish
        assert "es" in provider._get_voice_for_language("es")

        # French
        assert "fr" in provider._get_voice_for_language("fr")

        # Unknown language falls back to default
        unknown_voice = provider._get_voice_for_language("xyz")
        assert unknown_voice == provider.default_voice


class TestElevenLabsProvider:
    """Tests for ElevenLabsProvider."""

    def test_provider_name(self):
        """Test provider name property."""
        from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider(api_key="test")
        assert provider.name == "elevenlabs"

    def test_default_voices(self):
        """Test default voice mappings."""
        from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider

        assert "narrator" in ElevenLabsProvider.DEFAULT_VOICES
        assert "dramatic" in ElevenLabsProvider.DEFAULT_VOICES
        assert "energetic" in ElevenLabsProvider.DEFAULT_VOICES

    def test_provider_without_api_key(self):
        """Test provider initialization without API key."""
        from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider()
        # Should not raise, but will warn
        assert provider.api_key is None

    @pytest.mark.asyncio
    async def test_generate_without_api_key(self):
        """Test that generate fails gracefully without API key."""
        from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider()  # No API key
        request = VoiceoverRequest(text="Test")

        result = await provider.generate(request)

        assert result.success is False
        assert "not configured" in result.error_message.lower()
