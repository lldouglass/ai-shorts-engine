"""Unit tests for render pipeline tasks."""

from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.adapters.renderer.creatomate import SceneClip


class TestRenderPipelineHelpers:
    """Tests for render pipeline helper functions."""

    def test_get_voiceover_provider_stub(self):
        """Test that stub provider is returned by default."""
        with patch("shorts_engine.jobs.render_pipeline.settings") as mock_settings:
            mock_settings.voiceover_provider = "stub"

            from shorts_engine.adapters.voiceover.stub import StubVoiceoverProvider
            from shorts_engine.jobs.render_pipeline import get_voiceover_provider

            provider = get_voiceover_provider()
            assert isinstance(provider, StubVoiceoverProvider)

    def test_get_voiceover_provider_elevenlabs(self):
        """Test that ElevenLabs provider is returned when configured."""
        with patch("shorts_engine.jobs.render_pipeline.settings") as mock_settings:
            mock_settings.voiceover_provider = "elevenlabs"

            from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider
            from shorts_engine.jobs.render_pipeline import get_voiceover_provider

            provider = get_voiceover_provider()
            assert isinstance(provider, ElevenLabsProvider)

    def test_get_voiceover_provider_edge_tts(self):
        """Test that Edge TTS provider is returned when configured."""
        with patch("shorts_engine.jobs.render_pipeline.settings") as mock_settings:
            mock_settings.voiceover_provider = "edge_tts"

            from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider
            from shorts_engine.jobs.render_pipeline import get_voiceover_provider

            provider = get_voiceover_provider()
            assert isinstance(provider, EdgeTTSProvider)

    def test_get_renderer_provider_stub(self):
        """Test that stub renderer is returned by default."""
        with patch("shorts_engine.jobs.render_pipeline.settings") as mock_settings:
            mock_settings.renderer_provider = "stub"

            from shorts_engine.adapters.renderer.stub import StubRendererProvider
            from shorts_engine.jobs.render_pipeline import get_renderer_provider

            provider = get_renderer_provider()
            assert isinstance(provider, StubRendererProvider)

    def test_get_renderer_provider_creatomate(self):
        """Test that Creatomate provider is returned when configured."""
        with patch("shorts_engine.jobs.render_pipeline.settings") as mock_settings:
            mock_settings.renderer_provider = "creatomate"

            from shorts_engine.adapters.renderer.creatomate import CreatomateProvider
            from shorts_engine.jobs.render_pipeline import get_renderer_provider

            provider = get_renderer_provider()
            assert isinstance(provider, CreatomateProvider)


class TestSceneClipConstruction:
    """Tests for building scene clips from database models."""

    def test_scene_clip_from_data(self):
        """Test creating SceneClip from typical data."""
        clip = SceneClip(
            video_url="https://storage.example.com/clip_001.mp4",
            duration_seconds=5.5,
            caption_text="The beginning",
            scene_number=1,
        )

        assert clip.video_url == "https://storage.example.com/clip_001.mp4"
        assert clip.duration_seconds == 5.5
        assert clip.caption_text == "The beginning"
        assert clip.scene_number == 1

    def test_scene_clip_without_caption(self):
        """Test SceneClip without caption for no-caption renders."""
        clip = SceneClip(
            video_url="https://storage.example.com/clip_001.mp4",
            duration_seconds=5.0,
            caption_text=None,
            scene_number=1,
        )

        assert clip.caption_text is None

    def test_scene_clip_file_url(self):
        """Test SceneClip with file:// URL for local files."""
        clip = SceneClip(
            video_url="file:///storage/clips/scene1.mp4",
            duration_seconds=6.0,
            caption_text="Local file",
            scene_number=1,
        )

        assert clip.video_url.startswith("file://")


class TestNarrationScriptBuilding:
    """Tests for building narration scripts from caption beats."""

    def test_combine_caption_beats(self):
        """Test combining caption beats into narration."""
        caption_beats = [
            "The journey begins",
            "Darkness falls",
            "Hope emerges",
            "Victory awaits",
        ]

        narration = ". ".join(caption_beats)

        assert narration == "The journey begins. Darkness falls. Hope emerges. Victory awaits"

    def test_empty_caption_beats(self):
        """Test handling empty caption beats."""
        caption_beats = []
        narration = ". ".join(caption_beats)

        assert narration == ""
        assert not narration.strip()

    def test_filter_none_caption_beats(self):
        """Test filtering out None caption beats."""
        caption_beats = ["Start", None, "Middle", None, "End"]
        filtered = [cb for cb in caption_beats if cb]
        narration = ". ".join(filtered)

        assert narration == "Start. Middle. End"
        assert "None" not in narration


class TestRenderPipelineIntegration:
    """Integration tests for render pipeline (mocked external calls)."""

    @pytest.fixture
    def mock_scene(self):
        """Create a mock scene object."""
        scene = MagicMock()
        scene.id = "scene-123"
        scene.scene_number = 1
        scene.duration_seconds = 5.0
        scene.caption_beat = "Opening scene"
        return scene

    @pytest.fixture
    def mock_asset(self):
        """Create a mock asset object."""
        asset = MagicMock()
        asset.id = "asset-123"
        asset.url = "https://storage.example.com/clip.mp4"
        asset.file_path = None
        asset.status = "ready"
        return asset

    def test_scene_to_clip_conversion(self, mock_scene, mock_asset):
        """Test converting scene + asset to SceneClip."""
        clip = SceneClip(
            video_url=mock_asset.url or f"file://{mock_asset.file_path}",
            duration_seconds=mock_scene.duration_seconds,
            caption_text=mock_scene.caption_beat,
            scene_number=mock_scene.scene_number,
        )

        assert clip.video_url == "https://storage.example.com/clip.mp4"
        assert clip.duration_seconds == 5.0
        assert clip.caption_text == "Opening scene"
        assert clip.scene_number == 1


class TestVoiceoverTaskLogic:
    """Tests for voiceover generation task logic."""

    def test_narration_from_single_caption(self):
        """Test narration from a single caption beat."""
        captions = ["Epic moment"]
        narration = ". ".join(captions)

        assert narration == "Epic moment"

    def test_narration_length_affects_duration(self):
        """Test that longer narration estimates longer duration."""
        short_text = "Hi"
        long_text = "This is a much longer narration that should take more time to speak"

        # Rough estimate: 150 words per minute
        def estimate_duration(text):
            words = len(text.split())
            return (words / 150) * 60

        short_duration = estimate_duration(short_text)
        long_duration = estimate_duration(long_text)

        assert long_duration > short_duration


class TestRenderFinalVideoLogic:
    """Tests for final video rendering logic."""

    def test_total_duration_calculation(self):
        """Test calculating total duration from scene clips."""
        scenes = [
            SceneClip(video_url="url1", duration_seconds=5.0),
            SceneClip(video_url="url2", duration_seconds=6.0),
            SceneClip(video_url="url3", duration_seconds=4.5),
            SceneClip(video_url="url4", duration_seconds=5.5),
        ]

        total = sum(s.duration_seconds for s in scenes)
        assert total == 21.0

    def test_8_scene_typical_duration(self):
        """Test typical 8-scene video duration."""
        # Typical short: 7-8 scenes, 5-7 seconds each
        scenes = [SceneClip(video_url=f"url{i}", duration_seconds=5.0 + (i % 3)) for i in range(8)]

        total = sum(s.duration_seconds for s in scenes)
        # Should be roughly 40-56 seconds
        assert 40 <= total <= 60

    def test_voiceover_url_resolution(self):
        """Test resolving voiceover URL from different sources."""
        # From result
        result_url = {"success": True, "url": "https://voice.mp3"}
        voiceover_url = result_url.get("url") if result_url.get("success") else None
        assert voiceover_url == "https://voice.mp3"

        # Skipped result
        skipped_result = {"success": True, "skipped": True}
        voiceover_url = None
        if skipped_result.get("success") and not skipped_result.get("skipped"):
            voiceover_url = skipped_result.get("url")
        assert voiceover_url is None
