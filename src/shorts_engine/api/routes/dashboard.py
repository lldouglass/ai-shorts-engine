"""Dashboard endpoints for video performance analytics."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload

from shorts_engine.db.models import (
    PublishJobModel,
    VideoCommentModel,
    VideoJobModel,
    VideoMetricsModel,
    VideoRecipeFeaturesModel,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.logging import get_logger

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
logger = get_logger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class VideoPerformance(BaseModel):
    """Video performance summary."""

    publish_job_id: str
    video_job_id: str
    title: str | None
    platform: str
    platform_video_id: str | None
    platform_url: str | None
    published_at: datetime | None
    # Latest metrics
    views: int = 0
    likes: int = 0
    comments: int = 0
    engagement_rate: float | None = None
    avg_view_duration: float | None = None
    reward_score: float | None = None
    # Recipe features
    style_preset: str | None = None
    scene_count: int | None = None


class TopVideosResponse(BaseModel):
    """Response with top performing videos."""

    videos: list[VideoPerformance]
    total_count: int


class VideoMetricsDetail(BaseModel):
    """Detailed metrics for a video."""

    window_type: str
    window_start: datetime
    window_end: datetime
    views: int
    likes: int
    comments: int
    shares: int = 0
    watch_time_minutes: int = 0
    engagement_rate: float | None
    avg_view_duration: float | None
    avg_view_percentage: float | None = None
    reward_score: float | None
    fetched_at: datetime


class CommentSummary(BaseModel):
    """Comment summary."""

    id: str
    author: str | None
    text: str
    like_count: int
    reply_count: int = 0
    published_at: datetime | None
    sentiment: str | None = None


class DashboardStats(BaseModel):
    """Aggregate dashboard statistics."""

    total_published_videos: int
    total_views_24h: int
    total_views_7d: int
    average_reward_score: float | None
    videos_published_last_24h: int
    videos_published_last_7d: int
    top_style_preset: str | None = None


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/top-videos",
    response_model=TopVideosResponse,
    summary="Get top performing videos",
    description="List videos sorted by reward score or other metrics.",
)
async def get_top_videos(
    sort_by: str = Query(
        "reward_score",
        description="Sort metric: reward_score, views, engagement_rate, likes",
    ),
    window: str = Query(
        "24h",
        description="Time window for metrics: 1h, 6h, 24h, 72h, 7d",
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> TopVideosResponse:
    """Get top performing videos ranked by score."""
    valid_windows = {"1h", "6h", "24h", "72h", "7d"}
    if window not in valid_windows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid window. Must be one of: {', '.join(valid_windows)}",
        )

    valid_sort = {"reward_score", "views", "engagement_rate", "likes"}
    if sort_by not in valid_sort:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sort_by. Must be one of: {', '.join(valid_sort)}",
        )

    with get_session_context() as session:
        # Get total count of published videos
        total_count = (
            session.execute(
                select(func.count()).select_from(PublishJobModel).where(
                    PublishJobModel.status == "published"
                )
            ).scalar()
            or 0
        )

        # Get published videos with their latest metrics for the specified window
        # Using a subquery to get the latest metrics per publish_job
        latest_metrics_subq = (
            select(
                VideoMetricsModel.publish_job_id,
                VideoMetricsModel.views,
                VideoMetricsModel.likes,
                VideoMetricsModel.comments_count,
                VideoMetricsModel.engagement_rate,
                VideoMetricsModel.avg_view_duration_seconds,
                VideoMetricsModel.reward_score,
                VideoMetricsModel.fetched_at,
            )
            .where(VideoMetricsModel.window_type == window)
            .distinct(VideoMetricsModel.publish_job_id)
            .order_by(
                VideoMetricsModel.publish_job_id,
                desc(VideoMetricsModel.fetched_at),
            )
            .subquery()
        )

        # Main query with eager loading for recipe_features to avoid N+1
        query = (
            select(
                PublishJobModel,
                VideoJobModel,
                latest_metrics_subq.c.views,
                latest_metrics_subq.c.likes,
                latest_metrics_subq.c.comments_count,
                latest_metrics_subq.c.engagement_rate,
                latest_metrics_subq.c.avg_view_duration_seconds,
                latest_metrics_subq.c.reward_score,
            )
            .join(VideoJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .outerjoin(
                latest_metrics_subq,
                PublishJobModel.id == latest_metrics_subq.c.publish_job_id,
            )
            .options(selectinload(VideoJobModel.recipe_features))
            .where(PublishJobModel.status == "published")
        )

        # Apply sorting
        if sort_by == "reward_score":
            query = query.order_by(desc(latest_metrics_subq.c.reward_score).nullslast())
        elif sort_by == "views":
            query = query.order_by(desc(latest_metrics_subq.c.views).nullslast())
        elif sort_by == "engagement_rate":
            query = query.order_by(desc(latest_metrics_subq.c.engagement_rate).nullslast())
        elif sort_by == "likes":
            query = query.order_by(desc(latest_metrics_subq.c.likes).nullslast())

        # Pagination
        query = query.offset(offset).limit(limit)

        results = session.execute(query).all()

        videos = []
        for row in results:
            pub_job = row[0]
            video_job = row[1]

            # Use eager-loaded recipe_features (no extra query)
            features = video_job.recipe_features

            videos.append(
                VideoPerformance(
                    publish_job_id=str(pub_job.id),
                    video_job_id=str(video_job.id),
                    title=video_job.title,
                    platform=pub_job.platform,
                    platform_video_id=pub_job.platform_video_id,
                    platform_url=pub_job.platform_url,
                    published_at=pub_job.actual_publish_at,
                    views=row[2] or 0,
                    likes=row[3] or 0,
                    comments=row[4] or 0,
                    engagement_rate=row[5],
                    avg_view_duration=row[6],
                    reward_score=row[7],
                    style_preset=features.style_preset if features else video_job.style_preset,
                    scene_count=features.scene_count if features else None,
                )
            )

        return TopVideosResponse(videos=videos, total_count=total_count)


@router.get(
    "/video/{publish_job_id}/metrics",
    response_model=list[VideoMetricsDetail],
    summary="Get video metrics history",
    description="Get all metric windows for a specific video.",
)
async def get_video_metrics(publish_job_id: str) -> list[VideoMetricsDetail]:
    """Get metrics history for a video."""
    try:
        job_uuid = UUID(publish_job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid publish job ID",
        )

    with get_session_context() as session:
        # Verify publish job exists
        pub_job = session.get(PublishJobModel, job_uuid)
        if not pub_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Publish job not found",
            )

        metrics = (
            session.execute(
                select(VideoMetricsModel)
                .where(VideoMetricsModel.publish_job_id == job_uuid)
                .order_by(
                    VideoMetricsModel.window_type,
                    desc(VideoMetricsModel.fetched_at),
                )
            )
            .scalars()
            .all()
        )

        return [
            VideoMetricsDetail(
                window_type=m.window_type,
                window_start=m.window_start,
                window_end=m.window_end,
                views=m.views,
                likes=m.likes,
                comments=m.comments_count,
                shares=m.shares,
                watch_time_minutes=m.watch_time_minutes,
                engagement_rate=m.engagement_rate,
                avg_view_duration=m.avg_view_duration_seconds,
                avg_view_percentage=m.avg_view_percentage,
                reward_score=m.reward_score,
                fetched_at=m.fetched_at,
            )
            for m in metrics
        ]


@router.get(
    "/video/{publish_job_id}/comments",
    response_model=list[CommentSummary],
    summary="Get video comments",
    description="Get comments for a specific video.",
)
async def get_video_comments(
    publish_job_id: str,
    limit: int = Query(50, ge=1, le=200, description="Number of comments"),
    sort_by: str = Query("likes", description="Sort by: likes, date"),
) -> list[CommentSummary]:
    """Get comments for a video."""
    try:
        job_uuid = UUID(publish_job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid publish job ID",
        )

    with get_session_context() as session:
        # Verify publish job exists
        pub_job = session.get(PublishJobModel, job_uuid)
        if not pub_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Publish job not found",
            )

        query = select(VideoCommentModel).where(
            VideoCommentModel.publish_job_id == job_uuid
        )

        if sort_by == "likes":
            query = query.order_by(desc(VideoCommentModel.like_count))
        elif sort_by == "date":
            query = query.order_by(desc(VideoCommentModel.published_at).nullslast())
        else:
            query = query.order_by(desc(VideoCommentModel.like_count))

        query = query.limit(limit)

        comments = session.execute(query).scalars().all()

        return [
            CommentSummary(
                id=str(c.id),
                author=c.author_display_name,
                text=c.text,
                like_count=c.like_count,
                reply_count=c.reply_count,
                published_at=c.published_at,
                sentiment=c.sentiment,
            )
            for c in comments
        ]


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Get aggregate dashboard stats",
    description="Get high-level stats across all videos.",
)
async def get_dashboard_stats() -> DashboardStats:
    """Get aggregate stats for the dashboard."""
    with get_session_context() as session:
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)

        # Total published videos
        total_published = (
            session.execute(
                select(func.count()).select_from(PublishJobModel).where(
                    PublishJobModel.status == "published"
                )
            ).scalar()
            or 0
        )

        # Total views from 24h window metrics
        total_views_24h = (
            session.execute(
                select(func.sum(VideoMetricsModel.views)).where(
                    VideoMetricsModel.window_type == "24h"
                )
            ).scalar()
            or 0
        )

        # Total views from 7d window metrics
        total_views_7d = (
            session.execute(
                select(func.sum(VideoMetricsModel.views)).where(
                    VideoMetricsModel.window_type == "7d"
                )
            ).scalar()
            or 0
        )

        # Average reward score (from 24h window)
        avg_reward = session.execute(
            select(func.avg(VideoMetricsModel.reward_score)).where(
                VideoMetricsModel.window_type == "24h",
                VideoMetricsModel.reward_score.isnot(None),
            )
        ).scalar()

        # Videos published in last 24h
        videos_24h = (
            session.execute(
                select(func.count()).select_from(PublishJobModel).where(
                    PublishJobModel.status == "published",
                    PublishJobModel.actual_publish_at >= cutoff_24h,
                )
            ).scalar()
            or 0
        )

        # Videos published in last 7d
        videos_7d = (
            session.execute(
                select(func.count()).select_from(PublishJobModel).where(
                    PublishJobModel.status == "published",
                    PublishJobModel.actual_publish_at >= cutoff_7d,
                )
            ).scalar()
            or 0
        )

        # Top style preset (most used in last 7d)
        top_preset_result = session.execute(
            select(VideoJobModel.style_preset, func.count(VideoJobModel.id).label("count"))
            .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .where(
                PublishJobModel.status == "published",
                PublishJobModel.actual_publish_at >= cutoff_7d,
            )
            .group_by(VideoJobModel.style_preset)
            .order_by(desc("count"))
            .limit(1)
        ).first()

        top_preset = top_preset_result[0] if top_preset_result else None

        return DashboardStats(
            total_published_videos=total_published,
            total_views_24h=int(total_views_24h),
            total_views_7d=int(total_views_7d),
            average_reward_score=round(avg_reward, 4) if avg_reward else None,
            videos_published_last_24h=videos_24h,
            videos_published_last_7d=videos_7d,
            top_style_preset=top_preset,
        )


@router.get(
    "/video/{publish_job_id}",
    summary="Get video details with metrics",
    description="Get full details for a specific published video.",
)
async def get_video_details(publish_job_id: str) -> dict[str, Any]:
    """Get detailed information about a published video."""
    try:
        job_uuid = UUID(publish_job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid publish job ID",
        )

    with get_session_context() as session:
        # Fetch publish_job with eager-loaded video_job and recipe_features
        pub_job = session.execute(
            select(PublishJobModel)
            .options(
                selectinload(PublishJobModel.video_job).selectinload(VideoJobModel.recipe_features)
            )
            .where(PublishJobModel.id == job_uuid)
        ).scalar_one_or_none()

        if not pub_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Publish job not found",
            )

        video_job = pub_job.video_job

        # Fetch all metrics at once and group by window type (single query)
        all_metrics = session.execute(
            select(VideoMetricsModel)
            .where(VideoMetricsModel.publish_job_id == job_uuid)
            .order_by(VideoMetricsModel.window_type, desc(VideoMetricsModel.fetched_at))
        ).scalars().all()

        # Group by window type and take the latest (first after ordering)
        metrics_by_window = {}
        seen_windows: set[str] = set()
        for metric in all_metrics:
            if metric.window_type not in seen_windows:
                seen_windows.add(metric.window_type)
                metrics_by_window[metric.window_type] = {
                    "views": metric.views,
                    "likes": metric.likes,
                    "comments": metric.comments_count,
                    "engagement_rate": metric.engagement_rate,
                    "reward_score": metric.reward_score,
                    "fetched_at": metric.fetched_at.isoformat() if metric.fetched_at else None,
                }

        # Use eager-loaded recipe features
        features = video_job.recipe_features if video_job else None

        # Get comment count
        comment_count = (
            session.execute(
                select(func.count()).select_from(VideoCommentModel).where(
                    VideoCommentModel.publish_job_id == job_uuid
                )
            ).scalar()
            or 0
        )

        return {
            "publish_job": {
                "id": str(pub_job.id),
                "platform": pub_job.platform,
                "platform_video_id": pub_job.platform_video_id,
                "platform_url": pub_job.platform_url,
                "status": pub_job.status,
                "visibility": pub_job.visibility,
                "published_at": pub_job.actual_publish_at.isoformat() if pub_job.actual_publish_at else None,
            },
            "video_job": {
                "id": str(video_job.id) if video_job else None,
                "title": video_job.title if video_job else None,
                "description": video_job.description if video_job else None,
                "style_preset": video_job.style_preset if video_job else None,
                "idea": video_job.idea if video_job else None,
            },
            "metrics": metrics_by_window,
            "recipe_features": {
                "hook_type": features.hook_type if features else None,
                "scene_count": features.scene_count if features else None,
                "total_duration": features.total_duration_seconds if features else None,
                "caption_density": features.caption_density if features else None,
                "has_voiceover": features.has_voiceover if features else None,
                "narration_wpm": features.narration_wpm if features else None,
            } if features else None,
            "comment_count": comment_count,
        }
