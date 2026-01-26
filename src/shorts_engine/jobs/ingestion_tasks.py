"""Celery tasks for metrics and comments ingestion."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shorts_engine.adapters.analytics.youtube import YouTubeAnalyticsAdapter
from shorts_engine.adapters.analytics.instagram import InstagramAnalyticsAdapter
from shorts_engine.adapters.analytics.tiktok import TikTokAnalyticsAdapter
from shorts_engine.adapters.comments.youtube import YouTubeCommentsAdapter
from shorts_engine.adapters.comments.instagram import InstagramCommentsAdapter
from shorts_engine.adapters.comments.tiktok import TikTokCommentsAdapter
from shorts_engine.db.models import (
    PublishJobModel,
    VideoMetricsModel,
    VideoCommentModel,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.logging import get_logger
from shorts_engine.utils import run_async
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def compute_reward_score(
    views: int,
    likes: int,
    comments: int,
    shares: int,
    engagement_rate: float | None,
    avg_view_percentage: float | None,
) -> float:
    """Compute a reward score for RL optimization.

    Formula emphasizes:
    - Watch time / retention (primary driver of YouTube recommendations)
    - Engagement rate (likes + comments + shares / views)
    - View count (log scale to avoid outlier dominance)

    The score is normalized to [0, 1] range.

    Args:
        views: Total view count.
        likes: Total likes.
        comments: Total comments.
        shares: Total shares.
        engagement_rate: Pre-computed engagement rate (optional).
        avg_view_percentage: Average view percentage (0-100, optional).

    Returns:
        Reward score between 0 and 1.
    """
    if views == 0:
        return 0.0

    # Views component (log scale, cap at 1M)
    # log10(1) = 0, log10(1M) = 6
    views_score = min(math.log10(max(views, 1)) / 6, 1.0)

    # Engagement rate (typically 0-15%)
    # If not provided, compute from raw values
    if engagement_rate is None:
        engagement_rate = (likes + comments + shares) / views if views > 0 else 0
    engagement_score = min(engagement_rate / 0.15, 1.0)

    # Retention (avg view percentage, typically 20-60% for shorts)
    # Default to 30% if not available
    avg_pct = avg_view_percentage if avg_view_percentage is not None else 30
    retention_score = min(avg_pct / 60, 1.0)

    # Weighted combination
    # - Retention is weighted highest because watch time drives recommendations
    # - Engagement shows content quality
    # - Views are weighted lowest to avoid chasing viral but low-quality content
    reward = 0.3 * views_score + 0.3 * engagement_score + 0.4 * retention_score

    return round(reward, 4)


@celery_app.task(bind=True, name="ingest_metrics_batch")
def ingest_metrics_batch_task(
    self: Any,
    since_hours: int = 168,  # 7 days default
) -> dict[str, Any]:
    """Ingest metrics for all published videos.

    Iterates over all publish_jobs with platform_video_id and fetches
    metrics for each, upserting into video_metrics table.

    Args:
        since_hours: Only process videos published within this many hours.

    Returns:
        Result dict with counts.
    """
    task_id = self.request.id
    logger.info("ingest_metrics_batch_started", task_id=task_id, since_hours=since_hours)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    processed = 0
    metrics_created = 0
    errors = 0

    with get_session_context() as session:
        # Get all published videos with platform IDs
        publish_jobs = session.execute(
            select(PublishJobModel).where(
                PublishJobModel.platform_video_id.isnot(None),
                PublishJobModel.status == "published",
                PublishJobModel.actual_publish_at >= cutoff,
            )
        ).scalars().all()

        logger.info("ingest_metrics_batch_found", count=len(publish_jobs))

        for pub_job in publish_jobs:
            try:
                count = _ingest_single_video_metrics(session, pub_job)
                metrics_created += count
                processed += 1
            except Exception as e:
                logger.error(
                    "ingest_metrics_error",
                    publish_job_id=str(pub_job.id),
                    error=str(e),
                )
                errors += 1

    result = {
        "success": True,
        "task_id": task_id,
        "processed": processed,
        "metrics_created": metrics_created,
        "errors": errors,
    }
    logger.info("ingest_metrics_batch_completed", **result)
    return result


def _get_analytics_adapter(platform: str, account_id: UUID):
    """Get the appropriate analytics adapter for a platform.

    Args:
        platform: Platform name (youtube, instagram, tiktok).
        account_id: Account UUID for authentication.

    Returns:
        Analytics adapter instance.

    Raises:
        ValueError: If platform is not supported.
    """
    if platform == "youtube":
        return YouTubeAnalyticsAdapter(account_id=account_id)
    elif platform == "instagram":
        return InstagramAnalyticsAdapter(account_id=account_id)
    elif platform == "tiktok":
        return TikTokAnalyticsAdapter(account_id=account_id)
    else:
        raise ValueError(f"Unsupported platform for analytics: {platform}")


def _get_comments_adapter(platform: str, account_id: UUID):
    """Get the appropriate comments adapter for a platform.

    Args:
        platform: Platform name (youtube, instagram, tiktok).
        account_id: Account UUID for authentication.

    Returns:
        Comments adapter instance.

    Raises:
        ValueError: If platform is not supported.
    """
    if platform == "youtube":
        return YouTubeCommentsAdapter(account_id=account_id)
    elif platform == "instagram":
        return InstagramCommentsAdapter(account_id=account_id)
    elif platform == "tiktok":
        return TikTokCommentsAdapter(account_id=account_id)
    else:
        raise ValueError(f"Unsupported platform for comments: {platform}")


def _ingest_single_video_metrics(session, pub_job: PublishJobModel) -> int:
    """Ingest metrics for a single video.

    Args:
        session: Database session.
        pub_job: The publish job to ingest metrics for.

    Returns:
        Number of metric records created/updated.
    """
    supported_platforms = ("youtube", "instagram", "tiktok")
    if pub_job.platform not in supported_platforms:
        logger.debug("ingest_metrics_skip_platform", platform=pub_job.platform)
        return 0

    adapter = _get_analytics_adapter(pub_job.platform, pub_job.account_id)
    count = 0

    try:
        # Fetch windowed metrics
        publish_time = pub_job.actual_publish_at or pub_job.created_at
        windows = run_async(
            adapter.fetch_windowed_metrics(
                pub_job.platform_video_id,
                publish_time,
            )
        )

        for windowed in windows:
            metrics = windowed.metrics

            # Extract raw data fields
            raw_data = metrics.raw_data or {}
            dislikes = raw_data.get("dislikes", 0)
            avg_view_percentage = raw_data.get("avg_view_percentage")

            # Compute reward score
            reward_score = compute_reward_score(
                views=metrics.views,
                likes=metrics.likes,
                comments=metrics.comments_count,
                shares=metrics.shares,
                engagement_rate=metrics.engagement_rate,
                avg_view_percentage=avg_view_percentage,
            )

            # Upsert metric record
            stmt = pg_insert(VideoMetricsModel).values(
                publish_job_id=pub_job.id,
                window_type=windowed.window_type,
                window_start=windowed.window_start,
                window_end=windowed.window_end,
                views=metrics.views,
                likes=metrics.likes,
                dislikes=int(dislikes) if dislikes else 0,
                comments_count=metrics.comments_count,
                shares=metrics.shares,
                watch_time_minutes=metrics.watch_time_seconds // 60 if metrics.watch_time_seconds else 0,
                avg_view_duration_seconds=metrics.avg_view_duration_seconds,
                avg_view_percentage=avg_view_percentage,
                engagement_rate=metrics.engagement_rate,
                reward_score=reward_score,
                raw_data=metrics.raw_data,
                fetched_at=metrics.fetched_at,
            )

            # On conflict, update metrics
            stmt = stmt.on_conflict_do_update(
                constraint="uq_video_metrics_window",
                set_={
                    "views": stmt.excluded.views,
                    "likes": stmt.excluded.likes,
                    "dislikes": stmt.excluded.dislikes,
                    "comments_count": stmt.excluded.comments_count,
                    "shares": stmt.excluded.shares,
                    "watch_time_minutes": stmt.excluded.watch_time_minutes,
                    "avg_view_duration_seconds": stmt.excluded.avg_view_duration_seconds,
                    "avg_view_percentage": stmt.excluded.avg_view_percentage,
                    "engagement_rate": stmt.excluded.engagement_rate,
                    "reward_score": stmt.excluded.reward_score,
                    "raw_data": stmt.excluded.raw_data,
                    "fetched_at": stmt.excluded.fetched_at,
                },
            )

            session.execute(stmt)
            count += 1

        session.commit()

        logger.debug(
            "ingest_metrics_video_complete",
            publish_job_id=str(pub_job.id),
            video_id=pub_job.platform_video_id,
            windows=count,
        )

    finally:
        run_async(adapter.close())

    return count


@celery_app.task(bind=True, name="ingest_comments_batch")
def ingest_comments_batch_task(
    self: Any,
    since_hours: int = 168,
    max_per_video: int = 100,
) -> dict[str, Any]:
    """Ingest comments for all published videos.

    Args:
        since_hours: Only process videos published within this many hours.
        max_per_video: Max comments to fetch per video.

    Returns:
        Result dict with counts.
    """
    task_id = self.request.id
    logger.info(
        "ingest_comments_batch_started",
        task_id=task_id,
        since_hours=since_hours,
        max_per_video=max_per_video,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    processed = 0
    total_comments = 0
    errors = 0

    with get_session_context() as session:
        publish_jobs = session.execute(
            select(PublishJobModel).where(
                PublishJobModel.platform_video_id.isnot(None),
                PublishJobModel.status == "published",
                PublishJobModel.actual_publish_at >= cutoff,
            )
        ).scalars().all()

        logger.info("ingest_comments_batch_found", count=len(publish_jobs))

        for pub_job in publish_jobs:
            try:
                count = _ingest_single_video_comments(session, pub_job, max_per_video)
                total_comments += count
                processed += 1
            except Exception as e:
                logger.error(
                    "ingest_comments_error",
                    publish_job_id=str(pub_job.id),
                    error=str(e),
                )
                errors += 1

    result = {
        "success": True,
        "task_id": task_id,
        "processed": processed,
        "total_comments": total_comments,
        "errors": errors,
    }
    logger.info("ingest_comments_batch_completed", **result)
    return result


def _ingest_single_video_comments(
    session,
    pub_job: PublishJobModel,
    max_results: int,
) -> int:
    """Ingest comments for a single video.

    Args:
        session: Database session.
        pub_job: The publish job to ingest comments for.
        max_results: Maximum comments to fetch.

    Returns:
        Count of new/updated comments.
    """
    supported_platforms = ("youtube", "instagram", "tiktok")
    if pub_job.platform not in supported_platforms:
        logger.debug("ingest_comments_skip_platform", platform=pub_job.platform)
        return 0

    adapter = _get_comments_adapter(pub_job.platform, pub_job.account_id)
    new_count = 0

    try:
        comments = run_async(
            adapter.fetch_comments(
                pub_job.platform_video_id,
                max_results=max_results,
            )
        )

        for comment in comments:
            raw_data = comment.raw_data or {}

            # Upsert comment record
            stmt = pg_insert(VideoCommentModel).values(
                publish_job_id=pub_job.id,
                platform_comment_id=comment.platform_comment_id,
                author_channel_id=raw_data.get("author_channel_id"),
                author_display_name=comment.author,
                text=comment.text,
                like_count=comment.likes,
                reply_count=raw_data.get("reply_count", 0),
                published_at=comment.posted_at,
                raw_data=comment.raw_data,
                fetched_at=datetime.now(timezone.utc),
            )

            stmt = stmt.on_conflict_do_update(
                constraint="uq_video_comment_unique",
                set_={
                    "like_count": stmt.excluded.like_count,
                    "reply_count": stmt.excluded.reply_count,
                    "fetched_at": stmt.excluded.fetched_at,
                },
            )

            result = session.execute(stmt)
            # Count as new if actually inserted (not just updated)
            new_count += 1

        session.commit()

        logger.debug(
            "ingest_comments_video_complete",
            publish_job_id=str(pub_job.id),
            video_id=pub_job.platform_video_id,
            comments=new_count,
        )

    finally:
        run_async(adapter.close())

    return new_count


@celery_app.task(bind=True, name="ingest_single_video_metrics")
def ingest_single_video_metrics_task(
    self: Any,
    publish_job_id: str,
) -> dict[str, Any]:
    """Ingest metrics for a single video.

    Args:
        publish_job_id: UUID of the publish job.

    Returns:
        Result dict.
    """
    task_id = self.request.id
    logger.info(
        "ingest_single_video_metrics_started",
        task_id=task_id,
        publish_job_id=publish_job_id,
    )

    try:
        job_uuid = UUID(publish_job_id)
    except ValueError:
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid publish job ID",
        }

    with get_session_context() as session:
        pub_job = session.get(PublishJobModel, job_uuid)

        if not pub_job:
            return {
                "success": False,
                "task_id": task_id,
                "error": "Publish job not found",
            }

        if not pub_job.platform_video_id:
            return {
                "success": False,
                "task_id": task_id,
                "error": "No platform video ID",
            }

        try:
            count = _ingest_single_video_metrics(session, pub_job)
            return {
                "success": True,
                "task_id": task_id,
                "metrics_created": count,
            }
        except Exception as e:
            logger.error(
                "ingest_single_video_metrics_error",
                publish_job_id=publish_job_id,
                error=str(e),
            )
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
            }


@celery_app.task(bind=True, name="ingest_single_video_comments")
def ingest_single_video_comments_task(
    self: Any,
    publish_job_id: str,
    max_results: int = 100,
) -> dict[str, Any]:
    """Ingest comments for a single video.

    Args:
        publish_job_id: UUID of the publish job.
        max_results: Maximum comments to fetch.

    Returns:
        Result dict.
    """
    task_id = self.request.id
    logger.info(
        "ingest_single_video_comments_started",
        task_id=task_id,
        publish_job_id=publish_job_id,
    )

    try:
        job_uuid = UUID(publish_job_id)
    except ValueError:
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid publish job ID",
        }

    with get_session_context() as session:
        pub_job = session.get(PublishJobModel, job_uuid)

        if not pub_job:
            return {
                "success": False,
                "task_id": task_id,
                "error": "Publish job not found",
            }

        if not pub_job.platform_video_id:
            return {
                "success": False,
                "task_id": task_id,
                "error": "No platform video ID",
            }

        try:
            count = _ingest_single_video_comments(session, pub_job, max_results)
            return {
                "success": True,
                "task_id": task_id,
                "comments_ingested": count,
            }
        except Exception as e:
            logger.error(
                "ingest_single_video_comments_error",
                publish_job_id=publish_job_id,
                error=str(e),
            )
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
            }
