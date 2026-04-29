"""Tests for Seedance 2.0 via fal.ai."""

from unittest.mock import AsyncMock, call, patch

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
    async def test_generate_with_multiple_reference_images_uses_reference_to_video(self) -> None:
        provider = SeedanceProvider(api_key="test-key")
        reference_images = [
            b"\x89PNG\r\n\x1a\nprimary-ref",
            b"\x89PNG\r\n\x1a\nsecondary-ref",
        ]

        with (
            patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe,
            patch.object(
                provider, "_upload_reference_images", new_callable=AsyncMock
            ) as mock_uploads,
        ):
            mock_subscribe.return_value = {
                "video": {"url": "https://fal.media/files/example/seedance-ref2vid.mp4"}
            }
            mock_uploads.return_value = [
                "https://fal.media/uploads/ref-01.png",
                "https://fal.media/uploads/ref-02.png",
            ]

            request = VideoGenRequest(
                prompt="Animate this approved board into motion",
                duration_seconds=6,
                aspect_ratio="1:1",
                reference_images=reference_images,
                options={"seed": 314159},
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["model"] == SeedanceProvider.FAL_MODEL_REF2VID

        mock_subscribe.assert_called_once_with(
            SeedanceProvider.FAL_MODEL_REF2VID,
            arguments={
                "prompt": (
                    "@Image1 is the primary approved board / first-frame direction. "
                    "@Image2 is an additional visual continuity reference. "
                    "Animate this approved board into motion"
                ),
                "image_urls": [
                    "https://fal.media/uploads/ref-01.png",
                    "https://fal.media/uploads/ref-02.png",
                ],
                "duration": "6",
                "aspect_ratio": "1:1",
                "resolution": "720p",
                "generate_audio": False,
                "seed": 314159,
            },
        )
        mock_uploads.assert_called_once_with(reference_images)

    @pytest.mark.asyncio
    async def test_generate_with_end_reference_image_maps_to_end_image_url(self) -> None:
        provider = SeedanceProvider(api_key="test-key")
        start_image_bytes = b"\x89PNG\r\n\x1a\nstart-frame"
        end_image_bytes = b"\x89PNG\r\n\x1a\nend-frame"

        with (
            patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe,
            patch.object(
                provider, "_upload_reference_image", new_callable=AsyncMock
            ) as mock_upload,
        ):
            mock_subscribe.return_value = {
                "video": {"url": "https://fal.media/files/example/seedance-end-frame.mp4"},
                "seed": 17,
            }
            mock_upload.side_effect = [
                "https://fal.media/uploads/start-frame.png",
                "https://fal.media/uploads/end-frame.png",
            ]

            request = VideoGenRequest(
                prompt="Animate this approved board into motion",
                duration_seconds=6,
                aspect_ratio="1:1",
                reference_images=[start_image_bytes],
                options={
                    "seed": 17,
                    SeedanceProvider.END_REFERENCE_IMAGE_OPTION: end_image_bytes,
                },
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["model"] == SeedanceProvider.FAL_MODEL_IMG2VID
        assert result.metadata["seed"] == 17

        mock_subscribe.assert_called_once_with(
            SeedanceProvider.FAL_MODEL_IMG2VID,
            arguments={
                "prompt": "Animate this approved board into motion",
                "image_url": "https://fal.media/uploads/start-frame.png",
                "end_image_url": "https://fal.media/uploads/end-frame.png",
                "duration": "6",
                "aspect_ratio": "1:1",
                "resolution": "720p",
                "generate_audio": False,
                "seed": 17,
            },
        )
        assert mock_upload.await_args_list == [call(start_image_bytes), call(end_image_bytes)]

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
