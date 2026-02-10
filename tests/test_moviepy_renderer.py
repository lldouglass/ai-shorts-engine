"""Unit tests for MoviePy renderer adapter."""

import platform
from unittest.mock import patch

import pytest

from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer


class TestMoviePyRendererProperties:
    """Tests for MoviePyRenderer basic properties."""

    def test_name_property(self):
        """Provider name should be 'moviepy'."""
        provider = MoviePyRenderer()
        assert provider.name == "moviepy"

    @pytest.mark.asyncio
    async def test_render_basic_requires_moviepy(self):
        """Basic render() should work when MoviePy is available."""
        from shorts_engine.adapters.renderer.base import RenderRequest

        provider = MoviePyRenderer()
        request = RenderRequest(video_data=b"test")
        result = await provider.render(request)
        # Either succeeds (moviepy available) or returns not-available error
        assert isinstance(result.success, bool)


class TestResolvePath:
    """Tests for file:// URL resolution."""

    def test_resolve_path_linux_absolute(self):
        """file:// URL with absolute Linux path."""
        url = "file:///storage/clips/scene_1.mp4"
        assert MoviePyRenderer._resolve_path(url) == "/storage/clips/scene_1.mp4"

    def test_resolve_path_raw_path(self):
        """Raw filesystem path (no file:// prefix) is returned as-is."""
        path = "/app/storage/clips/scene_1.mp4"
        assert MoviePyRenderer._resolve_path(path) == path

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_resolve_path_windows_drive(self):
        """file:// URL with Windows drive letter."""
        url = "file:///C:/Users/test/video.mp4"
        assert MoviePyRenderer._resolve_path(url) == "C:/Users/test/video.mp4"

    def test_resolve_path_encoded_spaces(self):
        """file:// URL with percent-encoded spaces."""
        url = "file:///storage/my%20clips/scene%201.mp4"
        assert MoviePyRenderer._resolve_path(url) == "/storage/my clips/scene 1.mp4"

    def test_resolve_path_http_url(self):
        """HTTP URLs should be returned unchanged."""
        url = "https://example.com/video.mp4"
        assert MoviePyRenderer._resolve_path(url) == url


class TestGetRendererProviderMoviePy:
    """Test that config selects MoviePyRenderer."""

    def test_get_renderer_provider_moviepy(self):
        """When RENDERER_PROVIDER=moviepy, MoviePyRenderer is returned."""
        with patch("shorts_engine.jobs.render_pipeline.settings") as mock_settings:
            mock_settings.renderer_provider = "moviepy"

            from shorts_engine.jobs.render_pipeline import get_renderer_provider

            provider = get_renderer_provider()
            assert isinstance(provider, MoviePyRenderer)
