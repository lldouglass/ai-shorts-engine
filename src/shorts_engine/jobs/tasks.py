"""Celery task definitions for the video pipeline."""

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from shorts_engine.adapters.analytics.stub import StubAnalyticsAdapter
from shorts_engine.adapters.comments.stub import StubCommentsAdapter
from shorts_engine.adapters.publisher.stub import StubPublisherAdapter
from shorts_engine.adapters.renderer.stub import StubRendererProvider
from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def run_async(coro: Any) -> Any:
    """Run an async coroutine in a sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="smoke_test")
def smoke_test_task(self: Any) -> dict[str, Any]:
    """Smoke test task to verify the job queue is working.

    This task runs through a minimal version of the pipeline
    using stub adapters to verify everything is connected.
    """
    task_id = self.request.id
    logger.info("smoke_test_started", task_id=task_id)

    try:
        # Test video generation stub
        video_gen = StubVideoGenProvider()
        request = VideoGenRequest(prompt="Test video for smoke test", duration_seconds=10)
        gen_result = run_async(video_gen.generate(request))

        if not gen_result.success:
            raise RuntimeError(f"Video generation failed: {gen_result.error_message}")

        # Test renderer stub
        renderer = StubRendererProvider()
        from shorts_engine.adapters.renderer.base import RenderRequest

        render_request = RenderRequest(video_data=gen_result.video_data or b"")
        render_result = run_async(renderer.render(render_request))

        if not render_result.success:
            raise RuntimeError(f"Rendering failed: {render_result.error_message}")

        # Test publisher stub
        publisher = StubPublisherAdapter(platform=Platform.YOUTUBE)
        from shorts_engine.adapters.publisher.base import PublishRequest

        publish_request = PublishRequest(
            video_path=render_result.output_path,
            title="Smoke Test Video",
        )
        publish_result = run_async(publisher.publish(publish_request))

        if not publish_result.success:
            raise RuntimeError(f"Publishing failed: {publish_result.error_message}")

        # Test analytics stub
        analytics = StubAnalyticsAdapter(platform=Platform.YOUTUBE)
        metrics = run_async(analytics.fetch_metrics(publish_result.platform_video_id or ""))

        # Test comments stub
        comments = StubCommentsAdapter(platform=Platform.YOUTUBE)
        comment_list = run_async(
            comments.fetch_comments(publish_result.platform_video_id or "", max_results=5)
        )

        result = {
            "success": True,
            "task_id": task_id,
            "completed_at": datetime.now().isoformat(),
            "stages": {
                "video_gen": {"success": True, "provider": video_gen.name},
                "render": {
                    "success": True,
                    "provider": renderer.name,
                    "output_path": str(render_result.output_path),
                },
                "publish": {
                    "success": True,
                    "platform": publish_result.platform,
                    "video_id": publish_result.platform_video_id,
                },
                "analytics": {"success": True, "views": metrics.views},
                "comments": {"success": True, "count": len(comment_list)},
            },
        }

        logger.info("smoke_test_completed", **result)
        return result

    except Exception as e:
        logger.error("smoke_test_failed", task_id=task_id, error=str(e))
        return {
            "success": False,
            "task_id": task_id,
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        }


@celery_app.task(bind=True, name="generate_video")
def generate_video_task(
    self: Any,
    prompt: str,
    title: str | None = None,
    duration_seconds: int = 60,
) -> dict[str, Any]:
    """Generate a video from a prompt.

    Args:
        prompt: The text prompt for video generation
        title: Optional title for the video
        duration_seconds: Target video duration

    Returns:
        Result dict with video data or error
    """
    task_id = self.request.id
    logger.info(
        "generate_video_started",
        task_id=task_id,
        prompt=prompt[:100],
        duration=duration_seconds,
    )

    try:
        video_gen = StubVideoGenProvider()
        request = VideoGenRequest(
            prompt=prompt,
            duration_seconds=duration_seconds,
        )
        result = run_async(video_gen.generate(request))

        if result.success:
            logger.info("generate_video_completed", task_id=task_id)
            return {
                "success": True,
                "task_id": task_id,
                "title": title or f"Video {task_id[:8]}",
                "duration_seconds": result.duration_seconds,
                "metadata": result.metadata,
            }
        else:
            logger.error("generate_video_failed", task_id=task_id, error=result.error_message)
            return {
                "success": False,
                "task_id": task_id,
                "error": result.error_message,
            }

    except Exception as e:
        logger.error("generate_video_error", task_id=task_id, error=str(e))
        return {"success": False, "task_id": task_id, "error": str(e)}


@celery_app.task(bind=True, name="render_video")
def render_video_task(
    self: Any,
    video_id: str,
    output_format: str = "mp4",
) -> dict[str, Any]:
    """Render a generated video.

    Args:
        video_id: ID of the video to render
        output_format: Output format (mp4, webm, etc.)

    Returns:
        Result dict with output path or error
    """
    task_id = self.request.id
    logger.info("render_video_started", task_id=task_id, video_id=video_id)

    try:
        renderer = StubRendererProvider()
        from shorts_engine.adapters.renderer.base import RenderRequest

        # In real implementation, fetch video data from storage
        request = RenderRequest(
            video_data=b"placeholder_video_data",
            output_format=output_format,
        )
        result = run_async(renderer.render(request))

        if result.success:
            logger.info("render_video_completed", task_id=task_id, output=str(result.output_path))
            return {
                "success": True,
                "task_id": task_id,
                "video_id": video_id,
                "output_path": str(result.output_path),
                "file_size_bytes": result.file_size_bytes,
            }
        else:
            logger.error("render_video_failed", task_id=task_id, error=result.error_message)
            return {"success": False, "task_id": task_id, "error": result.error_message}

    except Exception as e:
        logger.error("render_video_error", task_id=task_id, error=str(e))
        return {"success": False, "task_id": task_id, "error": str(e)}


@celery_app.task(bind=True, name="publish_video")
def publish_video_task(
    self: Any,
    video_id: str,
    platform: str,
    title: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Publish a video to a platform.

    Args:
        video_id: ID of the video to publish
        platform: Target platform (youtube, tiktok, instagram)
        title: Video title
        description: Optional video description

    Returns:
        Result dict with platform video ID and URL or error
    """
    task_id = self.request.id
    logger.info("publish_video_started", task_id=task_id, video_id=video_id, platform=platform)

    try:
        platform_enum = Platform(platform.lower())
        publisher = StubPublisherAdapter(platform=platform_enum)

        from pathlib import Path

        from shorts_engine.adapters.publisher.base import PublishRequest

        # In real implementation, get actual video path from storage
        request = PublishRequest(
            video_path=Path("/tmp/placeholder.mp4"),
            title=title,
            description=description,
        )
        result = run_async(publisher.publish(request))

        if result.success:
            logger.info(
                "publish_video_completed",
                task_id=task_id,
                platform_video_id=result.platform_video_id,
            )
            return {
                "success": True,
                "task_id": task_id,
                "video_id": video_id,
                "platform": platform,
                "platform_video_id": result.platform_video_id,
                "url": result.url,
            }
        else:
            logger.error("publish_video_failed", task_id=task_id, error=result.error_message)
            return {"success": False, "task_id": task_id, "error": result.error_message}

    except Exception as e:
        logger.error("publish_video_error", task_id=task_id, error=str(e))
        return {"success": False, "task_id": task_id, "error": str(e)}


@celery_app.task(bind=True, name="ingest_analytics")
def ingest_analytics_task(
    self: Any,
    platform: str,
    platform_video_id: str,
) -> dict[str, Any]:
    """Ingest analytics for a published video.

    Args:
        platform: The platform (youtube, tiktok, instagram)
        platform_video_id: The video ID on the platform

    Returns:
        Result dict with metrics or error
    """
    task_id = self.request.id
    logger.info(
        "ingest_analytics_started",
        task_id=task_id,
        platform=platform,
        platform_video_id=platform_video_id,
    )

    try:
        platform_enum = Platform(platform.lower())
        analytics = StubAnalyticsAdapter(platform=platform_enum)
        metrics = run_async(analytics.fetch_metrics(platform_video_id))

        logger.info("ingest_analytics_completed", task_id=task_id, views=metrics.views)
        return {
            "success": True,
            "task_id": task_id,
            "platform": platform,
            "platform_video_id": platform_video_id,
            "metrics": {
                "views": metrics.views,
                "likes": metrics.likes,
                "comments_count": metrics.comments_count,
                "shares": metrics.shares,
                "engagement_rate": metrics.engagement_rate,
            },
            "fetched_at": metrics.fetched_at.isoformat(),
        }

    except Exception as e:
        logger.error("ingest_analytics_error", task_id=task_id, error=str(e))
        return {"success": False, "task_id": task_id, "error": str(e)}


@celery_app.task(bind=True, name="ingest_comments")
def ingest_comments_task(
    self: Any,
    platform: str,
    platform_video_id: str,
    max_results: int = 100,
) -> dict[str, Any]:
    """Ingest comments for a published video.

    Args:
        platform: The platform (youtube, tiktok, instagram)
        platform_video_id: The video ID on the platform
        max_results: Maximum number of comments to fetch

    Returns:
        Result dict with comments or error
    """
    task_id = self.request.id
    logger.info(
        "ingest_comments_started",
        task_id=task_id,
        platform=platform,
        platform_video_id=platform_video_id,
    )

    try:
        platform_enum = Platform(platform.lower())
        comments_adapter = StubCommentsAdapter(platform=platform_enum)
        comments = run_async(
            comments_adapter.fetch_comments(platform_video_id, max_results=max_results)
        )

        logger.info("ingest_comments_completed", task_id=task_id, count=len(comments))
        return {
            "success": True,
            "task_id": task_id,
            "platform": platform,
            "platform_video_id": platform_video_id,
            "comments_count": len(comments),
            "comments": [
                {
                    "id": c.platform_comment_id,
                    "author": c.author,
                    "text": c.text,
                    "likes": c.likes,
                }
                for c in comments[:10]  # Return first 10 in response
            ],
        }

    except Exception as e:
        logger.error("ingest_comments_error", task_id=task_id, error=str(e))
        return {"success": False, "task_id": task_id, "error": str(e)}
