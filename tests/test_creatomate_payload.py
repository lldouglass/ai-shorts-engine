"""Unit tests for Creatomate payload building."""


from shorts_engine.adapters.renderer.creatomate import (
    CreatomateProvider,
    CreatomateRenderRequest,
    SceneClip,
    build_creatomate_payload,
)


class TestSceneClip:
    """Tests for SceneClip dataclass."""

    def test_scene_clip_creation(self):
        """Test basic SceneClip creation."""
        clip = SceneClip(
            video_url="https://example.com/video.mp4",
            duration_seconds=5.0,
            caption_text="Test caption",
            scene_number=1,
        )
        assert clip.video_url == "https://example.com/video.mp4"
        assert clip.duration_seconds == 5.0
        assert clip.caption_text == "Test caption"
        assert clip.scene_number == 1

    def test_scene_clip_without_caption(self):
        """Test SceneClip without caption."""
        clip = SceneClip(
            video_url="https://example.com/video.mp4",
            duration_seconds=5.0,
        )
        assert clip.caption_text is None
        assert clip.scene_number == 0


class TestCreatomateRenderRequest:
    """Tests for CreatomateRenderRequest dataclass."""

    def test_request_defaults(self):
        """Test default values for render request."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=5.0),
        ]
        request = CreatomateRenderRequest(scenes=scenes)

        assert request.output_format == "mp4"
        assert request.width == 1080
        assert request.height == 1920
        assert request.fps == 30
        assert request.voiceover_url is None
        assert request.background_music_url is None
        assert request.background_music_volume == 0.3

    def test_request_with_all_options(self):
        """Test request with all options specified."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=5.0),
        ]
        request = CreatomateRenderRequest(
            scenes=scenes,
            voiceover_url="https://example.com/voice.mp3",
            background_music_url="https://example.com/music.mp3",
            background_music_volume=0.5,
            width=720,
            height=1280,
        )

        assert request.voiceover_url == "https://example.com/voice.mp3"
        assert request.background_music_url == "https://example.com/music.mp3"
        assert request.background_music_volume == 0.5
        assert request.width == 720
        assert request.height == 1280


class TestBuildCreatomatePayload:
    """Tests for the build_creatomate_payload function."""

    def test_basic_payload_structure(self):
        """Test that basic payload has required fields."""
        scenes = [
            SceneClip(
                video_url="https://example.com/scene1.mp4",
                duration_seconds=5.0,
                caption_text="Scene one",
                scene_number=1,
            ),
            SceneClip(
                video_url="https://example.com/scene2.mp4",
                duration_seconds=6.0,
                caption_text="Scene two",
                scene_number=2,
            ),
        ]

        payload = build_creatomate_payload(scenes)

        assert payload["output_format"] == "mp4"
        assert payload["width"] == 1080
        assert payload["height"] == 1920
        assert payload["frame_rate"] == 30
        assert payload["duration"] == 11.0  # 5 + 6 seconds
        assert "elements" in payload

    def test_video_elements_timing(self):
        """Test that video clips have correct timing."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=5.0),
            SceneClip(video_url="https://example.com/2.mp4", duration_seconds=6.0),
            SceneClip(video_url="https://example.com/3.mp4", duration_seconds=4.0),
        ]

        payload = build_creatomate_payload(scenes)
        elements = payload["elements"]

        # Find video elements
        video_elements = [e for e in elements if e.get("type") == "video"]
        assert len(video_elements) == 3

        # Check timing
        assert video_elements[0]["time"] == 0.0
        assert video_elements[0]["duration"] == 5.0
        assert video_elements[1]["time"] == 5.0
        assert video_elements[1]["duration"] == 6.0
        assert video_elements[2]["time"] == 11.0
        assert video_elements[2]["duration"] == 4.0

    def test_caption_elements(self):
        """Test that captions are created for scenes with caption_text."""
        scenes = [
            SceneClip(
                video_url="https://example.com/1.mp4",
                duration_seconds=5.0,
                caption_text="First caption",
            ),
            SceneClip(
                video_url="https://example.com/2.mp4",
                duration_seconds=5.0,
                caption_text=None,  # No caption
            ),
            SceneClip(
                video_url="https://example.com/3.mp4",
                duration_seconds=5.0,
                caption_text="Third caption",
            ),
        ]

        payload = build_creatomate_payload(scenes)
        elements = payload["elements"]

        # Find text elements
        text_elements = [e for e in elements if e.get("type") == "text"]
        assert len(text_elements) == 2

        # Check captions
        assert text_elements[0]["text"] == "First caption"
        assert text_elements[0]["time"] == 0.0
        assert text_elements[1]["text"] == "Third caption"
        assert text_elements[1]["time"] == 10.0  # After first two scenes

    def test_voiceover_element(self):
        """Test that voiceover audio track is added."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=5.0),
        ]

        payload = build_creatomate_payload(
            scenes,
            voiceover_url="https://example.com/voiceover.mp3",
        )
        elements = payload["elements"]

        # Find audio elements
        audio_elements = [e for e in elements if e.get("type") == "audio"]
        assert len(audio_elements) == 1
        assert audio_elements[0]["source"] == "https://example.com/voiceover.mp3"
        assert audio_elements[0]["volume"] == "100%"

    def test_background_music_element(self):
        """Test that background music is added with lower volume."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=5.0),
        ]

        payload = build_creatomate_payload(
            scenes,
            background_music_url="https://example.com/music.mp3",
        )
        elements = payload["elements"]

        audio_elements = [e for e in elements if e.get("type") == "audio"]
        assert len(audio_elements) == 1
        assert audio_elements[0]["source"] == "https://example.com/music.mp3"
        assert audio_elements[0]["volume"] == "30%"  # Default 0.3

    def test_both_audio_tracks(self):
        """Test payload with both voiceover and background music."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=10.0),
        ]

        payload = build_creatomate_payload(
            scenes,
            voiceover_url="https://example.com/voice.mp3",
            background_music_url="https://example.com/music.mp3",
        )
        elements = payload["elements"]

        audio_elements = [e for e in elements if e.get("type") == "audio"]
        assert len(audio_elements) == 2

        # Voiceover should be at 100%
        voiceover = next(e for e in audio_elements if "voice" in e["source"])
        assert voiceover["volume"] == "100%"

        # Music should be at 30%
        music = next(e for e in audio_elements if "music" in e["source"])
        assert music["volume"] == "30%"
        assert music.get("audio_fade_out") == 2.0

    def test_custom_caption_style(self):
        """Test that custom caption style is applied."""
        scenes = [
            SceneClip(
                video_url="https://example.com/1.mp4",
                duration_seconds=5.0,
                caption_text="Styled caption",
            ),
        ]

        custom_style = {
            "font_family": "Arial",
            "font_size": "10 vmin",
            "fill_color": "#ff0000",
        }

        payload = build_creatomate_payload(scenes, caption_style=custom_style)
        elements = payload["elements"]

        text_elements = [e for e in elements if e.get("type") == "text"]
        assert len(text_elements) == 1
        assert text_elements[0]["font_family"] == "Arial"
        assert text_elements[0]["font_size"] == "10 vmin"
        assert text_elements[0]["fill_color"] == "#ff0000"

    def test_custom_dimensions(self):
        """Test custom width and height."""
        scenes = [
            SceneClip(video_url="https://example.com/1.mp4", duration_seconds=5.0),
        ]

        payload = build_creatomate_payload(scenes, width=720, height=1280)

        assert payload["width"] == 720
        assert payload["height"] == 1280

    def test_8_scenes_payload(self):
        """Test payload with 8 scenes (typical short video)."""
        scenes = [
            SceneClip(
                video_url=f"https://example.com/scene{i}.mp4",
                duration_seconds=5.0 + (i % 3),
                caption_text=f"Caption {i}",
                scene_number=i,
            )
            for i in range(1, 9)
        ]

        payload = build_creatomate_payload(
            scenes,
            voiceover_url="https://example.com/voice.mp3",
        )

        # Total duration: 5+6+7+5+6+7+5+6 = 47 seconds
        expected_duration = sum(5.0 + (i % 3) for i in range(1, 9))
        assert payload["duration"] == expected_duration

        elements = payload["elements"]

        # Should have 8 video elements
        video_elements = [e for e in elements if e.get("type") == "video"]
        assert len(video_elements) == 8

        # Should have 8 caption elements
        text_elements = [e for e in elements if e.get("type") == "text"]
        assert len(text_elements) == 8

        # Should have 1 audio element (voiceover)
        audio_elements = [e for e in elements if e.get("type") == "audio"]
        assert len(audio_elements) == 1


class TestCreatomateProvider:
    """Tests for CreatomateProvider class."""

    def test_provider_name(self):
        """Test provider name property."""
        provider = CreatomateProvider(api_key="test")
        assert provider.name == "creatomate"

    def test_provider_without_api_key(self):
        """Test provider initialization without API key."""
        provider = CreatomateProvider()
        assert provider.api_key is None

    def test_default_caption_style(self):
        """Test default caption style generation."""
        provider = CreatomateProvider(api_key="test")
        style = provider._default_caption_style()

        assert "font_family" in style
        assert "font_size" in style
        assert "fill_color" in style
        assert style["y"] == "85%"  # Near bottom
        assert style["text_transform"] == "uppercase"

    def test_build_composition_payload_method(self):
        """Test the internal payload building method."""
        provider = CreatomateProvider(api_key="test")

        scenes = [
            SceneClip(
                video_url="https://example.com/1.mp4",
                duration_seconds=5.0,
                caption_text="Test",
            ),
        ]

        request = CreatomateRenderRequest(
            scenes=scenes,
            width=1080,
            height=1920,
        )

        payload = provider._build_composition_payload(request)

        assert payload["width"] == 1080
        assert payload["height"] == 1920
        assert payload["duration"] == 5.0
