"""Job management endpoints."""

from typing import Any
from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from shorts_engine.jobs.tasks import (
    generate_video_task,
    ingest_analytics_task,
    ingest_comments_task,
    publish_video_task,
    smoke_test_task,
)
from shorts_engine.jobs.render_pipeline import run_render_pipeline_task
from shorts_engine.jobs.video_pipeline import run_full_pipeline_task
from shorts_engine.jobs.publish_pipeline import run_publish_pipeline_task
from shorts_engine.logging import get_logger
from shorts_engine.worker import celery_app

router = APIRouter(prefix="/jobs", tags=["Jobs"])
logger = get_logger(__name__)


class JobResponse(BaseModel):
    """Response when a job is enqueued."""

    task_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response with job status details."""

    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class GenerateVideoRequest(BaseModel):
    """Request to generate a video."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    title: str | None = Field(None, max_length=255)
    duration_seconds: int = Field(default=60, ge=10, le=180)


class PublishVideoRequest(BaseModel):
    """Request to publish a video."""

    video_id: str
    platform: str = Field(..., pattern="^(youtube|tiktok|instagram)$")
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)


class IngestRequest(BaseModel):
    """Request to ingest analytics or comments."""

    platform: str = Field(..., pattern="^(youtube|tiktok|instagram)$")
    platform_video_id: str = Field(..., min_length=1)


class CreateShortRequest(BaseModel):
    """Request to create a short video."""

    project_id: str = Field(..., description="Project UUID")
    idea: str = Field(..., min_length=10, max_length=5000, description="Video idea/concept")
    style_preset: str = Field(
        default="DARK_DYSTOPIAN_ANIME",
        description="Style preset name",
    )


class RenderVideoRequest(BaseModel):
    """Request to render a final video."""

    video_job_id: str = Field(..., description="Video job UUID to render")
    include_voiceover: bool = Field(default=True, description="Generate voiceover")
    include_captions: bool = Field(default=True, description="Burn in captions")
    voice_id: str | None = Field(None, description="Voice ID for voiceover")
    narration_script: str | None = Field(None, description="Custom narration script")
    background_music_url: str | None = Field(None, description="Background music URL")


@router.post(
    "/smoke",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run smoke test",
    description="Enqueue a smoke test job to verify the pipeline is working.",
)
async def trigger_smoke_test() -> JobResponse:
    """Trigger a smoke test job."""
    logger.info("smoke_test_triggered")

    task = smoke_test_task.delay()

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Smoke test job enqueued successfully",
    )


@router.post(
    "/generate",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate video",
    description="Enqueue a video generation job.",
)
async def trigger_generate_video(request: GenerateVideoRequest) -> JobResponse:
    """Trigger a video generation job."""
    logger.info("generate_video_triggered", prompt=request.prompt[:50])

    task = generate_video_task.delay(
        prompt=request.prompt,
        title=request.title,
        duration_seconds=request.duration_seconds,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Video generation job enqueued successfully",
    )


@router.post(
    "/publish",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Publish video",
    description="Enqueue a video publishing job.",
)
async def trigger_publish_video(request: PublishVideoRequest) -> JobResponse:
    """Trigger a video publishing job."""
    logger.info(
        "publish_video_triggered",
        video_id=request.video_id,
        platform=request.platform,
    )

    task = publish_video_task.delay(
        video_id=request.video_id,
        platform=request.platform,
        title=request.title,
        description=request.description,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Video publishing job enqueued successfully",
    )


@router.post(
    "/ingest/analytics",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest analytics",
    description="Enqueue an analytics ingestion job.",
)
async def trigger_ingest_analytics(request: IngestRequest) -> JobResponse:
    """Trigger an analytics ingestion job."""
    logger.info(
        "ingest_analytics_triggered",
        platform=request.platform,
        platform_video_id=request.platform_video_id,
    )

    task = ingest_analytics_task.delay(
        platform=request.platform,
        platform_video_id=request.platform_video_id,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Analytics ingestion job enqueued successfully",
    )


@router.post(
    "/ingest/comments",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest comments",
    description="Enqueue a comments ingestion job.",
)
async def trigger_ingest_comments(request: IngestRequest) -> JobResponse:
    """Trigger a comments ingestion job."""
    logger.info(
        "ingest_comments_triggered",
        platform=request.platform,
        platform_video_id=request.platform_video_id,
    )

    task = ingest_comments_task.delay(
        platform=request.platform,
        platform_video_id=request.platform_video_id,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Comments ingestion job enqueued successfully",
    )


@router.get(
    "/{task_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Get the status and result of a job by task ID.",
)
async def get_job_status(task_id: str) -> JobStatusResponse:
    """Get the status of a job."""
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return JobStatusResponse(
            task_id=task_id,
            status="pending",
        )
    elif result.state == "STARTED":
        return JobStatusResponse(
            task_id=task_id,
            status="running",
        )
    elif result.state == "SUCCESS":
        return JobStatusResponse(
            task_id=task_id,
            status="completed",
            result=result.result,
        )
    elif result.state == "FAILURE":
        return JobStatusResponse(
            task_id=task_id,
            status="failed",
            error=str(result.result),
        )
    else:
        return JobStatusResponse(
            task_id=task_id,
            status=result.state.lower(),
        )


@router.post(
    "/shorts/create",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create short video",
    description="Enqueue the full video creation pipeline (plan -> generate -> verify -> ready).",
)
async def trigger_create_short(request: CreateShortRequest) -> JobResponse:
    """Trigger the video creation pipeline."""
    logger.info(
        "create_short_triggered",
        project_id=request.project_id,
        style_preset=request.style_preset,
    )

    task = run_full_pipeline_task.delay(
        project_id=request.project_id,
        idea=request.idea,
        style_preset=request.style_preset,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Video creation pipeline enqueued successfully",
    )


@router.post(
    "/shorts/render",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Render final video",
    description="Enqueue the render pipeline (voiceover -> compose -> ready to publish).",
)
async def trigger_render_video(request: RenderVideoRequest) -> JobResponse:
    """Trigger the render pipeline for a video job."""
    logger.info(
        "render_video_triggered",
        video_job_id=request.video_job_id,
        include_voiceover=request.include_voiceover,
    )

    task = run_render_pipeline_task.delay(
        video_job_id=request.video_job_id,
        include_voiceover=request.include_voiceover,
        include_captions=request.include_captions,
        voice_id=request.voice_id,
        narration_script=request.narration_script,
        background_music_url=request.background_music_url,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Render pipeline enqueued successfully",
    )


class PublishShortRequest(BaseModel):
    """Request to publish a short video to platforms."""

    video_job_id: str = Field(..., description="Video job UUID to publish")
    youtube_account: str | None = Field(None, description="YouTube account label")
    scheduled_publish_at: str | None = Field(None, description="ISO 8601 datetime for scheduled publishing")
    visibility: str = Field(default="public", pattern="^(public|private|unlisted)$")
    dry_run: bool = Field(default=False, description="Log what would happen without uploading")


class PublishStatusResponse(BaseModel):
    """Response with publish job details."""

    publish_job_id: str
    status: str
    platform: str
    platform_video_id: str | None = None
    platform_url: str | None = None
    scheduled_publish_at: str | None = None
    visibility: str | None = None
    forced_private: bool = False
    error_message: str | None = None
    dry_run: bool = False


@router.post(
    "/shorts/publish",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Publish short video",
    description="Enqueue the publish pipeline to upload a rendered video to platforms.",
)
async def trigger_publish_short(request: PublishShortRequest) -> JobResponse:
    """Trigger the publish pipeline for a video job."""
    if not request.youtube_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one platform account must be specified (youtube_account)",
        )

    logger.info(
        "publish_short_triggered",
        video_job_id=request.video_job_id,
        youtube_account=request.youtube_account,
        dry_run=request.dry_run,
    )

    task = run_publish_pipeline_task.delay(
        video_job_id=request.video_job_id,
        youtube_account=request.youtube_account,
        scheduled_publish_at=request.scheduled_publish_at,
        visibility=request.visibility,
        dry_run=request.dry_run,
    )

    return JobResponse(
        task_id=task.id,
        status="queued",
        message="Publish pipeline enqueued successfully",
    )


@router.get(
    "/publish/{publish_job_id}",
    response_model=PublishStatusResponse,
    summary="Get publish job status",
    description="Get the status of a specific publish job.",
)
async def get_publish_job_status(publish_job_id: str) -> PublishStatusResponse:
    """Get the status of a publish job."""
    from shorts_engine.db.models import PublishJobModel
    from shorts_engine.db.session import get_session_context

    try:
        job_uuid = UUID(publish_job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid publish job ID",
        )

    with get_session_context() as session:
        publish_job = session.get(PublishJobModel, job_uuid)

        if not publish_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Publish job not found",
            )

        return PublishStatusResponse(
            publish_job_id=str(publish_job.id),
            status=publish_job.status,
            platform=publish_job.platform,
            platform_video_id=publish_job.platform_video_id,
            platform_url=publish_job.platform_url,
            scheduled_publish_at=(
                publish_job.scheduled_publish_at.isoformat()
                if publish_job.scheduled_publish_at
                else None
            ),
            visibility=publish_job.visibility,
            forced_private=publish_job.forced_private,
            error_message=publish_job.error_message,
            dry_run=publish_job.dry_run,
        )
