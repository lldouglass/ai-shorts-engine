"""Tests for adapter implementations."""

from pathlib import Path

import pytest

from shorts_engine.adapters.publisher.base import PublishRequest
from shorts_engine.adapters.renderer.base import RenderRequest
from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.domain.enums import Platform


@pytest.mark.asyncio
async def test_video_gen_stub(video_gen_provider) -> None:
    """Test stub video generation provider."""
    request = VideoGenRequest(
        prompt="Test prompt for video generation",
        duration_seconds=30,
    )

    result = await video_gen_provider.generate(request)

    assert result.success is True
    assert result.video_data is not None
    assert len(result.video_data) > 0
    assert result.duration_seconds == 30.0
    assert result.metadata is not None


@pytest.mark.asyncio
async def test_video_gen_health_check(video_gen_provider) -> None:
    """Test video gen provider health check."""
    is_healthy = await video_gen_provider.health_check()
    assert is_healthy is True


@pytest.mark.asyncio
async def test_renderer_stub(renderer_provider) -> None:
    """Test stub renderer provider."""
    request = RenderRequest(
        video_data=b"test video data",
        output_format="mp4",
    )

    result = await renderer_provider.render(request)

    assert result.success is True
    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.file_size_bytes is not None
    assert result.file_size_bytes > 0


@pytest.mark.asyncio
async def test_publisher_stub(publisher_adapter) -> None:
    """Test stub publisher adapter."""
    import tempfile

    # Create a temp file
    temp_file = Path(tempfile.mktemp(suffix=".mp4"))
    temp_file.write_bytes(b"test video content")

    try:
        request = PublishRequest(
            video_path=temp_file,
            title="Test Video",
            description="Test description",
        )

        result = await publisher_adapter.publish(request)

        assert result.success is True
        assert result.platform == Platform.YOUTUBE
        assert result.platform_video_id is not None
        assert result.url is not None
    finally:
        temp_file.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_analytics_stub(analytics_adapter) -> None:
    """Test stub analytics adapter."""
    metrics = await analytics_adapter.fetch_metrics("test_video_123")

    assert metrics.platform == Platform.YOUTUBE
    assert metrics.platform_video_id == "test_video_123"
    assert metrics.views >= 0
    assert metrics.likes >= 0
    assert metrics.fetched_at is not None


@pytest.mark.asyncio
async def test_comments_stub(comments_adapter) -> None:
    """Test stub comments adapter."""
    comments = await comments_adapter.fetch_comments("test_video_123", max_results=10)

    assert isinstance(comments, list)
    assert len(comments) <= 10

    if comments:
        comment = comments[0]
        assert comment.platform == Platform.YOUTUBE
        assert comment.platform_video_id == "test_video_123"
        assert comment.text is not None
