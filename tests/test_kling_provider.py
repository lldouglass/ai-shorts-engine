"""Tests for Kling 2.6 Pro video generation provider."""

from unittest.mock import AsyncMock, patch

import pytest

from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.kling import KlingProvider


class TestKlingProviderName:
    def test_name(self) -> None:
        provider = KlingProvider(api_key="test-key")
        assert provider.name == "kling"


class TestDurationMapping:
    @pytest.mark.parametrize(
        ("input_seconds", "expected"),
        [
            (1, "5"),
            (3, "5"),
            (5, "5"),
            (6, "10"),
            (8, "10"),
            (10, "10"),
        ],
    )
    def test_map_duration(self, input_seconds: int, expected: str) -> None:
        assert KlingProvider._map_duration(input_seconds) == expected


class TestKlingGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        provider = KlingProvider(api_key="test-key")

        mock_result = {
            "video": {
                "url": "https://fal.media/files/example/output.mp4",
            }
        }

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.return_value = mock_result

            request = VideoGenRequest(
                prompt="A cat walking through a forest",
                duration_seconds=5,
                aspect_ratio="9:16",
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["video_url"] == "https://fal.media/files/example/output.mp4"
        assert result.metadata["provider"] == "kling"
        assert result.metadata["model"] == KlingProvider.FAL_MODEL
        assert result.duration_seconds == 5.0

        mock_subscribe.assert_called_once_with(
            KlingProvider.FAL_MODEL,
            arguments={
                "prompt": "A cat walking through a forest",
                "duration": "5",
                "aspect_ratio": "9:16",
                "negative_prompt": "blur, distort, low quality",
                "generate_audio": False,
            },
        )

    @pytest.mark.asyncio
    async def test_generate_with_style(self) -> None:
        provider = KlingProvider(api_key="test-key")

        mock_result = {"video": {"url": "https://fal.media/files/example/output.mp4"}}

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.return_value = mock_result

            request = VideoGenRequest(
                prompt="A cat walking",
                duration_seconds=8,
                style="cinematic noir",
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.duration_seconds == 10.0  # 8 > 5 → mapped to "10"

        call_args = mock_subscribe.call_args[1]["arguments"]
        assert call_args["prompt"] == "cinematic noir, A cat walking"
        assert call_args["duration"] == "10"

    @pytest.mark.asyncio
    async def test_generate_with_custom_negative_prompt(self) -> None:
        provider = KlingProvider(api_key="test-key")

        mock_result = {"video": {"url": "https://fal.media/files/example/output.mp4"}}

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.return_value = mock_result

            request = VideoGenRequest(
                prompt="A scene",
                duration_seconds=5,
                negative_prompt="text, watermark",
            )
            await provider.generate(request)

        call_args = mock_subscribe.call_args[1]["arguments"]
        assert call_args["negative_prompt"] == "text, watermark"

    @pytest.mark.asyncio
    async def test_generate_no_video_url(self) -> None:
        provider = KlingProvider(api_key="test-key")

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.return_value = {"video": {}}

            request = VideoGenRequest(prompt="A scene", duration_seconds=5)
            result = await provider.generate(request)

        assert result.success is False
        assert "no video URL" in result.error_message

    @pytest.mark.asyncio
    async def test_generate_fal_error(self) -> None:
        provider = KlingProvider(api_key="test-key")

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.side_effect = Exception("fal API rate limit exceeded")

            request = VideoGenRequest(prompt="A scene", duration_seconds=5)
            result = await provider.generate(request)

        assert result.success is False
        assert "fal API rate limit exceeded" in result.error_message

    @pytest.mark.asyncio
    async def test_generate_no_api_key(self) -> None:
        provider = KlingProvider(api_key=None)
        # Clear env var if set
        import os

        env_backup = os.environ.pop("FAL_KEY", None)
        try:
            request = VideoGenRequest(prompt="A scene", duration_seconds=5)
            result = await provider.generate(request)

            assert result.success is False
            assert "FAL_KEY not configured" in result.error_message
        finally:
            if env_backup:
                os.environ["FAL_KEY"] = env_backup


class TestKlingHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_with_key(self) -> None:
        provider = KlingProvider(api_key="test-key")
        assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_without_key(self) -> None:
        import os

        env_backup = os.environ.pop("FAL_KEY", None)
        try:
            provider = KlingProvider(api_key=None)
            assert await provider.health_check() is False
        finally:
            if env_backup:
                os.environ["FAL_KEY"] = env_backup


class TestKlingCheckStatus:
    @pytest.mark.asyncio
    async def test_check_status(self) -> None:
        provider = KlingProvider(api_key="test-key")

        with patch("fal_client.status_async", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = "COMPLETED"

            result = await provider.check_status("req-123")

        assert result["request_id"] == "req-123"
        assert result["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_check_status_error(self) -> None:
        provider = KlingProvider(api_key="test-key")

        with patch("fal_client.status_async", new_callable=AsyncMock) as mock_status:
            mock_status.side_effect = Exception("Not found")

            result = await provider.check_status("req-bad")

        assert "error" in result
        assert "Not found" in result["error"]
