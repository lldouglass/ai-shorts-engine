"""Tests for Seedance 2.0 via fal.ai."""

from unittest.mock import AsyncMock, patch

import pytest

from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.seedance import SeedanceProvider


class TestSeedanceProvider:
    def test_name(self) -> None:
        provider = SeedanceProvider(api_key="test-key")
        assert provider.name == "seedance"
        assert provider.supports_reference_images is True

    @pytest.mark.parametrize(
        ("input_seconds", "expected"),
        [
            (1, "4"),
            (4, "4"),
            (7, "7"),
            (15, "15"),
            (20, "15"),
        ],
    )
    def test_map_duration(self, input_seconds: int, expected: str) -> None:
        assert SeedanceProvider._map_duration(input_seconds) == expected

    @pytest.mark.asyncio
    async def test_generate_text_to_video_success(self) -> None:
        provider = SeedanceProvider(api_key="test-key")

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.return_value = {
                "video": {"url": "https://fal.media/files/example/seedance.mp4"},
                "seed": 42,
                "request_id": "req-seedance-001",
            }

            request = VideoGenRequest(
                prompt="A cat walking through a forest",
                duration_seconds=5,
                aspect_ratio="9:16",
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["provider"] == "seedance"
        assert result.metadata["model"] == SeedanceProvider.FAL_MODEL
        assert result.metadata["video_url"] == "https://fal.media/files/example/seedance.mp4"
        assert result.metadata["request_id"] == "req-seedance-001"
        assert result.duration_seconds == 5.0

        mock_subscribe.assert_called_once_with(
            SeedanceProvider.FAL_MODEL,
            arguments={
                "prompt": "A cat walking through a forest",
                "duration": "5",
                "aspect_ratio": "9:16",
                "resolution": "720p",
                "generate_audio": False,
            },
        )

    @pytest.mark.asyncio
    async def test_generate_with_reference_image_uses_image_to_video(self) -> None:
        provider = SeedanceProvider(api_key="test-key")
        fake_image_bytes = b"\x89PNG\r\n\x1a\nfake-png-data"

        with (
            patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe,
            patch.object(
                provider, "_upload_reference_image", new_callable=AsyncMock
            ) as mock_upload,
        ):
            mock_subscribe.return_value = {
                "video": {"url": "https://fal.media/files/example/seedance-img2vid.mp4"}
            }
            mock_upload.return_value = "https://fal.media/uploads/approved-board.png"

            request = VideoGenRequest(
                prompt="Animate this approved board into motion",
                duration_seconds=6,
                aspect_ratio="1:1",
                reference_images=[fake_image_bytes],
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["model"] == SeedanceProvider.FAL_MODEL_IMG2VID

        mock_subscribe.assert_called_once_with(
            SeedanceProvider.FAL_MODEL_IMG2VID,
            arguments={
                "prompt": "Animate this approved board into motion",
                "image_url": "https://fal.media/uploads/approved-board.png",
                "duration": "6",
                "aspect_ratio": "1:1",
                "resolution": "720p",
                "generate_audio": False,
            },
        )
        mock_upload.assert_called_once_with(fake_image_bytes)

    @pytest.mark.asyncio
    async def test_generate_without_api_key_fails(self) -> None:
        import os

        env_backup = os.environ.pop("FAL_KEY", None)
        try:
            provider = SeedanceProvider(api_key=None)
            result = await provider.generate(VideoGenRequest(prompt="A scene", duration_seconds=5))
            assert result.success is False
            assert "FAL_KEY not configured" in (result.error_message or "")
        finally:
            if env_backup:
                os.environ["FAL_KEY"] = env_backup
