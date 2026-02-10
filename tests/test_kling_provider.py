"""Tests for Kling video generation provider (text-to-video and image-to-video)."""

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


class TestKlingImageToVideo:
    @pytest.mark.asyncio
    async def test_generate_with_reference_image(self) -> None:
        """When reference_images is provided, use image-to-video model."""
        provider = KlingProvider(api_key="test-key")

        mock_result = {
            "video": {"url": "https://fal.media/files/example/img2vid_output.mp4"}
        }
        fake_image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-data"

        with (
            patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe,
            patch.object(
                provider, "_upload_reference_image", new_callable=AsyncMock
            ) as mock_upload,
        ):
            mock_subscribe.return_value = mock_result
            mock_upload.return_value = "https://fal.media/uploads/ref.jpg"

            request = VideoGenRequest(
                prompt="A forest path continuing",
                duration_seconds=5,
                aspect_ratio="9:16",
                reference_images=[fake_image_bytes],
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["model"] == KlingProvider.FAL_MODEL_IMG2VID
        assert result.metadata["video_url"] == "https://fal.media/files/example/img2vid_output.mp4"

        # Verify image-to-video model was called with image_url
        mock_subscribe.assert_called_once_with(
            KlingProvider.FAL_MODEL_IMG2VID,
            arguments={
                "prompt": "A forest path continuing",
                "image_url": "https://fal.media/uploads/ref.jpg",
                "duration": "5",
                "aspect_ratio": "9:16",
                "negative_prompt": "blur, distort, low quality",
                "generate_audio": False,
            },
        )
        mock_upload.assert_called_once_with(fake_image_bytes)

    @pytest.mark.asyncio
    async def test_generate_without_reference_uses_text_to_video(self) -> None:
        """Without reference_images, use text-to-video model (unchanged behavior)."""
        provider = KlingProvider(api_key="test-key")

        mock_result = {
            "video": {"url": "https://fal.media/files/example/output.mp4"}
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
        assert result.metadata["model"] == KlingProvider.FAL_MODEL

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
    async def test_generate_empty_reference_images_uses_text_to_video(self) -> None:
        """Empty reference_images list should still use text-to-video."""
        provider = KlingProvider(api_key="test-key")

        mock_result = {
            "video": {"url": "https://fal.media/files/example/output.mp4"}
        }

        with patch("fal_client.subscribe_async", new_callable=AsyncMock) as mock_subscribe:
            mock_subscribe.return_value = mock_result

            request = VideoGenRequest(
                prompt="A scene",
                duration_seconds=5,
                reference_images=[],
            )
            result = await provider.generate(request)

        assert result.success is True
        assert result.metadata["model"] == KlingProvider.FAL_MODEL


class TestKlingUploadReferenceImage:
    @pytest.mark.asyncio
    async def test_upload_reference_image(self) -> None:
        """Test uploading reference image to fal CDN."""
        provider = KlingProvider(api_key="test-key")
        fake_image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-data"

        with patch(
            "fal_client.upload_file_async", new_callable=AsyncMock
        ) as mock_upload:
            mock_upload.return_value = "https://fal.media/uploads/test.jpg"

            url = await provider._upload_reference_image(fake_image_bytes)

        assert url == "https://fal.media/uploads/test.jpg"
        mock_upload.assert_called_once()
        # Verify it was called with a temp file path (string)
        call_arg = mock_upload.call_args[0][0]
        assert call_arg.endswith(".jpg")

    @pytest.mark.asyncio
    async def test_upload_reference_image_cleans_up_temp_file(self) -> None:
        """Temp file should be cleaned up even on success."""
        import os

        provider = KlingProvider(api_key="test-key")
        fake_image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-data"
        captured_path = None

        async def capture_path(path: str) -> str:
            nonlocal captured_path
            captured_path = path
            return "https://fal.media/uploads/test.jpg"

        with patch("fal_client.upload_file_async", side_effect=capture_path):
            await provider._upload_reference_image(fake_image_bytes)

        assert captured_path is not None
        assert not os.path.exists(captured_path), "Temp file should be deleted after upload"

    @pytest.mark.asyncio
    async def test_upload_reference_image_cleans_up_on_error(self) -> None:
        """Temp file should be cleaned up even if upload fails."""
        import os

        provider = KlingProvider(api_key="test-key")
        fake_image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-data"
        captured_path = None

        async def capture_and_fail(path: str) -> str:
            nonlocal captured_path
            captured_path = path
            raise RuntimeError("Upload failed")

        with (
            patch("fal_client.upload_file_async", side_effect=capture_and_fail),
            pytest.raises(RuntimeError, match="Upload failed"),
        ):
            await provider._upload_reference_image(fake_image_bytes)

        assert captured_path is not None
        assert not os.path.exists(captured_path), "Temp file should be deleted after failure"


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
