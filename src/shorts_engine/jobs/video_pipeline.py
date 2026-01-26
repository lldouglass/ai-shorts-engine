"""Video creation pipeline Celery tasks.

Pipeline stages:
1. plan_job - Generate video plan using LLM
2. generate_scene_clips - Generate video clips for each scene
3. verify_assets - Verify all assets are ready
4. mark_ready_for_render - Mark job as ready for rendering

All tasks support:
- Retries with exponential backoff
- Idempotency via database checks
- Proper error handling and status updates
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from celery import chain, group
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select
from sqlalchemy.orm import Session

from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.luma import LumaProvider
from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider
from shorts_engine.config import settings
from shorts_engine.db.models import AssetModel, ProjectModel, PromptModel, SceneModel, VideoJobModel
from shorts_engine.db.session import get_session_context
from shorts_engine.logging import get_logger
from shorts_engine.presets.styles import get_preset
from shorts_engine.services.planner import PlannerService
from shorts_engine.services.storage import StorageService
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def run_async(coro: Any) -> Any:
    """Run an async coroutine in a sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def generate_idempotency_key(project_id: str, idea: str, preset: str) -> str:
    """Generate idempotency key from job parameters."""
    content = f"{project_id}:{idea}:{preset}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def get_video_gen_provider():
    """Get the configured video generation provider."""
    provider = getattr(settings, "video_gen_provider", "stub").lower()

    if provider == "luma":
        return LumaProvider()
    else:
        return StubVideoGenProvider()


# =============================================================================
# PIPELINE STAGE 1: Plan Job
# =============================================================================


@celery_app.task(
    bind=True,
    name="pipeline.plan_job",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
)
def plan_job_task(
    self,
    video_job_id: str,
) -> dict[str, Any]:
    """Generate a video plan using the LLM planner.

    Args:
        video_job_id: UUID of the video job to plan

    Returns:
        Dict with plan details and scene IDs
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info("plan_job_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        # Load job
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Idempotency check - skip if already planned
        if job.stage in ("planned", "generating", "verifying", "ready"):
            logger.info("plan_job_already_done", video_job_id=video_job_id, stage=job.stage)
            scene_ids = [str(s.id) for s in job.scenes]
            return {
                "success": True,
                "video_job_id": video_job_id,
                "stage": job.stage,
                "scene_ids": scene_ids,
                "skipped": True,
            }

        # Update status
        job.status = "running"
        job.stage = "planning"
        job.celery_task_id = task_id
        job.started_at = datetime.now(timezone.utc)
        session.commit()

        try:
            # Run planner
            planner = PlannerService()
            plan = run_async(planner.plan(job.idea, job.style_preset))

            # Update job with plan
            job.title = plan.title
            job.description = plan.description
            job.plan_data = plan.raw_response

            # Create scenes
            scene_ids = []
            preset = get_preset(job.style_preset)

            for scene_plan in plan.scenes:
                scene = SceneModel(
                    id=uuid4(),
                    video_job_id=job.id,
                    scene_number=scene_plan.scene_number,
                    visual_prompt=scene_plan.visual_prompt,
                    continuity_notes=scene_plan.continuity_notes,
                    caption_beat=scene_plan.caption_beat,
                    duration_seconds=scene_plan.duration_seconds,
                    status="pending",
                )
                session.add(scene)
                scene_ids.append(str(scene.id))

                # Store the prompt
                prompt = PromptModel(
                    id=uuid4(),
                    scene_id=scene.id,
                    prompt_type="visual",
                    prompt_text=scene_plan.visual_prompt,
                    model=planner.llm.name,
                    is_final=True,
                )
                session.add(prompt)

            job.stage = "planned"
            session.commit()

            logger.info(
                "plan_job_completed",
                video_job_id=video_job_id,
                title=plan.title,
                scene_count=len(scene_ids),
            )

            return {
                "success": True,
                "video_job_id": video_job_id,
                "title": plan.title,
                "scene_ids": scene_ids,
                "total_duration": plan.total_duration,
            }

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.retry_count = (job.retry_count or 0) + 1
            session.commit()
            raise


# =============================================================================
# PIPELINE STAGE 2: Generate Scene Clips
# =============================================================================


@celery_app.task(
    bind=True,
    name="pipeline.generate_scene_clip",
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
)
def generate_scene_clip_task(
    self,
    scene_id: str,
    video_job_id: str,
) -> dict[str, Any]:
    """Generate a video clip for a single scene.

    Args:
        scene_id: UUID of the scene to generate
        video_job_id: UUID of the parent video job

    Returns:
        Dict with asset information
    """
    task_id = self.request.id
    scene_uuid = UUID(scene_id)
    job_uuid = UUID(video_job_id)

    logger.info(
        "generate_scene_clip_started",
        task_id=task_id,
        scene_id=scene_id,
        video_job_id=video_job_id,
    )

    with get_session_context() as session:
        # Load scene
        scene = session.get(SceneModel, scene_uuid)
        if not scene:
            raise ValueError(f"Scene not found: {scene_id}")

        # Load job for style info
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Idempotency check - skip if already generated
        existing_asset = session.execute(
            select(AssetModel).where(
                AssetModel.scene_id == scene_uuid,
                AssetModel.asset_type == "scene_clip",
                AssetModel.status == "ready",
            )
        ).scalar_one_or_none()

        if existing_asset:
            logger.info("generate_scene_clip_already_done", scene_id=scene_id)
            return {
                "success": True,
                "scene_id": scene_id,
                "asset_id": str(existing_asset.id),
                "skipped": True,
            }

        # Update scene status
        scene.status = "generating"
        scene.generation_attempts = (scene.generation_attempts or 0) + 1
        session.commit()

        try:
            # Get style preset for prompt enhancement
            preset = get_preset(job.style_preset)
            style_suffix = preset.format_style_prompt() if preset else ""

            # Build full prompt
            full_prompt = scene.visual_prompt
            if style_suffix:
                full_prompt = f"{full_prompt}, {style_suffix}"
            if scene.continuity_notes:
                full_prompt = f"{full_prompt}. {scene.continuity_notes}"

            # Generate video
            provider = get_video_gen_provider()
            request = VideoGenRequest(
                prompt=full_prompt,
                duration_seconds=int(scene.duration_seconds),
                aspect_ratio="9:16",  # Vertical for shorts
            )

            result = run_async(provider.generate(request))

            if not result.success:
                scene.status = "failed"
                scene.last_error = result.error_message
                session.commit()
                raise RuntimeError(f"Video generation failed: {result.error_message}")

            # Store asset
            storage = StorageService()

            # Check if we have a URL or raw bytes
            video_url = result.metadata.get("video_url") if result.metadata else None

            if video_url:
                # Download and store from URL
                stored = run_async(
                    storage.store_from_url(
                        url=video_url,
                        asset_type="scene_clip",
                        video_job_id=job_uuid,
                        scene_id=scene_uuid,
                        metadata={
                            "provider": provider.name,
                            "generation_id": result.metadata.get("generation_id"),
                        },
                    )
                )
            elif result.video_data:
                # Store raw bytes
                stored = run_async(
                    storage.store_bytes(
                        data=result.video_data,
                        asset_type="scene_clip",
                        video_job_id=job_uuid,
                        scene_id=scene_uuid,
                        metadata={"provider": provider.name},
                    )
                )
            else:
                # URL reference only (stub or external storage)
                stored = storage.store_url_reference(
                    url=video_url or "stub://generated",
                    asset_type="scene_clip",
                    video_job_id=job_uuid,
                    scene_id=scene_uuid,
                    metadata=result.metadata or {},
                )

            # Create asset record
            asset = AssetModel(
                id=stored.id,
                video_job_id=job_uuid,
                scene_id=scene_uuid,
                asset_type="scene_clip",
                storage_type=stored.storage_type,
                file_path=str(stored.file_path) if stored.file_path else None,
                url=stored.url,
                external_id=result.metadata.get("generation_id") if result.metadata else None,
                provider=provider.name,
                file_size_bytes=stored.file_size_bytes,
                duration_seconds=result.duration_seconds,
                mime_type=stored.mime_type,
                width=1080,
                height=1920,
                status="ready",
                metadata_=stored.metadata,
            )
            session.add(asset)

            # Update scene
            scene.status = "generated"
            session.commit()

            logger.info(
                "generate_scene_clip_completed",
                scene_id=scene_id,
                asset_id=str(asset.id),
                provider=provider.name,
            )

            return {
                "success": True,
                "scene_id": scene_id,
                "asset_id": str(asset.id),
                "storage_type": stored.storage_type,
                "file_path": str(stored.file_path) if stored.file_path else None,
                "url": stored.url,
            }

        except Exception as e:
            scene.status = "failed"
            scene.last_error = str(e)
            session.commit()
            raise


@celery_app.task(
    bind=True,
    name="pipeline.generate_all_scenes",
)
def generate_all_scenes_task(
    self,
    video_job_id: str,
    scene_ids: list[str],
) -> dict[str, Any]:
    """Orchestrate generation of all scene clips.

    This task dispatches individual scene generation tasks
    and waits for all to complete.
    """
    logger.info(
        "generate_all_scenes_started",
        video_job_id=video_job_id,
        scene_count=len(scene_ids),
    )

    with get_session_context() as session:
        job = session.get(VideoJobModel, UUID(video_job_id))
        if job:
            job.stage = "generating"
            session.commit()

    # Create a group of scene generation tasks
    generation_tasks = group(
        generate_scene_clip_task.s(scene_id, video_job_id) for scene_id in scene_ids
    )

    # Execute and wait for all
    result = generation_tasks.apply_async()

    # Wait for completion (with timeout)
    try:
        results = result.get(timeout=3600)  # 1 hour max

        success_count = sum(1 for r in results if r.get("success"))
        logger.info(
            "generate_all_scenes_completed",
            video_job_id=video_job_id,
            success_count=success_count,
            total_count=len(scene_ids),
        )

        return {
            "success": success_count == len(scene_ids),
            "video_job_id": video_job_id,
            "results": results,
            "success_count": success_count,
            "total_count": len(scene_ids),
        }

    except Exception as e:
        logger.error(
            "generate_all_scenes_failed",
            video_job_id=video_job_id,
            error=str(e),
        )
        return {
            "success": False,
            "video_job_id": video_job_id,
            "error": str(e),
        }


# =============================================================================
# PIPELINE STAGE 3: Verify Assets
# =============================================================================


@celery_app.task(
    bind=True,
    name="pipeline.verify_assets",
    max_retries=3,
    default_retry_delay=30,
)
def verify_assets_task(
    self,
    video_job_id: str,
) -> dict[str, Any]:
    """Verify all scene assets are ready.

    Args:
        video_job_id: UUID of the video job

    Returns:
        Dict with verification results
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info("verify_assets_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        job.stage = "verifying"
        session.commit()

        # Check all scenes have assets
        scenes = session.execute(
            select(SceneModel).where(SceneModel.video_job_id == job_uuid).order_by(
                SceneModel.scene_number
            )
        ).scalars().all()

        storage = StorageService()
        verified_assets = []
        failed_assets = []

        for scene in scenes:
            asset = session.execute(
                select(AssetModel).where(
                    AssetModel.scene_id == scene.id,
                    AssetModel.asset_type == "scene_clip",
                )
            ).scalar_one_or_none()

            if not asset:
                failed_assets.append({
                    "scene_id": str(scene.id),
                    "scene_number": scene.scene_number,
                    "error": "No asset found",
                })
                continue

            if asset.status != "ready":
                failed_assets.append({
                    "scene_id": str(scene.id),
                    "scene_number": scene.scene_number,
                    "error": f"Asset status: {asset.status}",
                })
                continue

            # Verify asset is accessible
            from shorts_engine.services.storage import StoredAsset

            stored = StoredAsset(
                id=asset.id,
                storage_type=asset.storage_type,
                file_path=Path(asset.file_path) if asset.file_path else None,
                url=asset.url,
                file_size_bytes=asset.file_size_bytes,
                mime_type=asset.mime_type,
                checksum=None,
                metadata=asset.metadata_ or {},
                created_at=asset.created_at,
            )

            is_valid = run_async(storage.verify_asset(stored))

            if is_valid:
                verified_assets.append({
                    "scene_id": str(scene.id),
                    "scene_number": scene.scene_number,
                    "asset_id": str(asset.id),
                    "url": asset.url or f"file://{asset.file_path}",
                })
            else:
                failed_assets.append({
                    "scene_id": str(scene.id),
                    "scene_number": scene.scene_number,
                    "error": "Asset verification failed",
                })

        all_verified = len(failed_assets) == 0 and len(verified_assets) == len(scenes)

        logger.info(
            "verify_assets_completed",
            video_job_id=video_job_id,
            verified_count=len(verified_assets),
            failed_count=len(failed_assets),
            all_verified=all_verified,
        )

        return {
            "success": all_verified,
            "video_job_id": video_job_id,
            "verified_assets": verified_assets,
            "failed_assets": failed_assets,
        }


# =============================================================================
# PIPELINE STAGE 4: Mark Ready for Render
# =============================================================================


@celery_app.task(
    bind=True,
    name="pipeline.mark_ready_for_render",
)
def mark_ready_for_render_task(
    self,
    video_job_id: str,
    verification_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark a video job as ready for rendering.

    Args:
        video_job_id: UUID of the video job
        verification_result: Optional result from verify_assets

    Returns:
        Dict with final status
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info("mark_ready_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Check verification result
        if verification_result and not verification_result.get("success"):
            job.status = "failed"
            job.stage = "verification_failed"
            job.error_message = f"Asset verification failed: {verification_result.get('failed_assets')}"
            session.commit()

            return {
                "success": False,
                "video_job_id": video_job_id,
                "stage": job.stage,
                "error": job.error_message,
            }

        # Mark as ready
        job.status = "completed"
        job.stage = "ready"
        job.completed_at = datetime.now(timezone.utc)
        session.commit()

        # Collect final asset URLs
        assets = session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == job_uuid,
                AssetModel.asset_type == "scene_clip",
                AssetModel.status == "ready",
            )
        ).scalars().all()

        asset_urls = [
            {
                "scene_id": str(a.scene_id),
                "url": a.url or f"file://{a.file_path}",
                "duration": a.duration_seconds,
            }
            for a in assets
        ]

        logger.info(
            "mark_ready_completed",
            video_job_id=video_job_id,
            asset_count=len(asset_urls),
        )

        return {
            "success": True,
            "video_job_id": video_job_id,
            "stage": "ready",
            "title": job.title,
            "description": job.description,
            "assets": asset_urls,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }


# =============================================================================
# FULL PIPELINE ORCHESTRATOR
# =============================================================================


@celery_app.task(
    bind=True,
    name="pipeline.run_full_pipeline",
)
def run_full_pipeline_task(
    self,
    project_id: str,
    idea: str,
    style_preset: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Run the full video creation pipeline.

    Stages:
    1. Create/find video job
    2. Plan video with LLM
    3. Generate scene clips
    4. Verify assets
    5. Mark ready for render

    Args:
        project_id: UUID of the project
        idea: One-paragraph video idea
        style_preset: Style preset name
        idempotency_key: Optional idempotency key

    Returns:
        Dict with job ID and status
    """
    task_id = self.request.id
    project_uuid = UUID(project_id)

    # Generate idempotency key if not provided
    if not idempotency_key:
        idempotency_key = generate_idempotency_key(project_id, idea, style_preset)

    logger.info(
        "full_pipeline_started",
        task_id=task_id,
        project_id=project_id,
        idempotency_key=idempotency_key,
    )

    with get_session_context() as session:
        # Check for existing job with same idempotency key
        existing_job = session.execute(
            select(VideoJobModel).where(VideoJobModel.idempotency_key == idempotency_key)
        ).scalar_one_or_none()

        if existing_job:
            logger.info(
                "full_pipeline_existing_job",
                video_job_id=str(existing_job.id),
                stage=existing_job.stage,
            )
            video_job_id = str(existing_job.id)
        else:
            # Create new job
            job = VideoJobModel(
                id=uuid4(),
                project_id=project_uuid,
                idempotency_key=idempotency_key,
                idea=idea,
                style_preset=style_preset,
                status="pending",
                stage="created",
            )
            session.add(job)
            session.commit()
            video_job_id = str(job.id)
            logger.info("full_pipeline_job_created", video_job_id=video_job_id)

    # Build pipeline chain
    pipeline = chain(
        plan_job_task.s(video_job_id),
        # After planning, generate all scenes
        generate_all_scenes_from_plan.s(video_job_id),
        # Verify all assets
        verify_assets_task.s(),
        # Mark ready
        mark_ready_for_render_task.s(),
    )

    # Start pipeline
    result = pipeline.apply_async()

    return {
        "success": True,
        "video_job_id": video_job_id,
        "pipeline_task_id": result.id,
        "idempotency_key": idempotency_key,
    }


@celery_app.task(
    bind=True,
    name="pipeline.generate_all_scenes_from_plan",
)
def generate_all_scenes_from_plan(
    self,
    plan_result: dict[str, Any],
    video_job_id: str,
) -> str:
    """Extract scene IDs from plan result and trigger generation.

    This is a bridge task between planning and generation stages.
    """
    if not plan_result.get("success"):
        raise RuntimeError(f"Planning failed: {plan_result.get('error', 'Unknown error')}")

    scene_ids = plan_result.get("scene_ids", [])
    if not scene_ids:
        raise RuntimeError("No scenes in plan result")

    # Dispatch generation for all scenes
    result = generate_all_scenes_task.apply_async(
        args=[video_job_id, scene_ids]
    )

    # Wait for completion
    gen_result = result.get(timeout=3600)

    # Return video_job_id for next stage
    return video_job_id
