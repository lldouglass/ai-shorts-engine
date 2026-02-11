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

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from celery import chain
from sqlalchemy import select

from shorts_engine.adapters.video_gen.base import VideoGenProvider, VideoGenRequest
from shorts_engine.adapters.video_gen.luma import LumaProvider
from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider
from shorts_engine.config import settings
from shorts_engine.db.models import AssetModel, PromptModel, SceneModel, StoryModel, VideoJobModel
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import QACheckType, QAStage, QAStatus
from shorts_engine.logging import get_logger
from shorts_engine.presets.styles import get_preset
from shorts_engine.services.alerting import alert_pipeline_failure, alert_qa_failure
from shorts_engine.services.learning.context import OptimizationContextBuilder
from shorts_engine.services.metrics import (
    estimate_planning_cost,
    record_cost_estimate,
    record_job_completed,
    record_job_failed,
    record_job_started,
    record_qa_check,
)
from shorts_engine.services.planner import PlannerService
from shorts_engine.services.qa import QAFailedException, QAService
from shorts_engine.services.storage import StorageService
from shorts_engine.utils import run_async
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def generate_idempotency_key(project_id: str, idea: str, preset: str) -> str:
    """Generate idempotency key from job parameters."""
    content = f"{project_id}:{idea}:{preset}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def get_video_gen_provider() -> VideoGenProvider:
    """Get the configured video generation provider."""
    provider = getattr(settings, "video_gen_provider", "stub").lower()

    if provider == "luma":
        return LumaProvider()
    elif provider == "veo":
        from shorts_engine.adapters.video_gen.veo import VeoProvider

        return VeoProvider(model=settings.veo_model)
    elif provider == "kling":
        from shorts_engine.adapters.video_gen.kling import KlingProvider

        return KlingProvider()
    else:
        return StubVideoGenProvider()


# =============================================================================
# PIPELINE STAGE 1: Plan Job
# =============================================================================


@celery_app.task(
    bind=True,
    name="pipeline.plan_job",
    max_retries=5,  # Increased for QA retries
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
)
def plan_job_task(
    self: Any,  # noqa: ARG001 - used for self.request.id
    video_job_id: str,
) -> dict[str, Any]:
    """Generate a video plan using the LLM planner.

    Includes QA validation with retry logic:
    - On QA fail with attempts < max: regenerate plan
    - On QA fail with attempts exhausted: mark as FAILED_QA

    Args:
        video_job_id: UUID of the video job to plan

    Returns:
        Dict with plan details and scene IDs
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)
    start_time = datetime.now(UTC)

    logger.info("plan_job_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        # Load job
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Record job started metric
        record_job_started(session, job_uuid, "planning")

        # Idempotency check - skip if already planned and QA passed
        if job.stage in ("planned", "generating", "verifying", "ready") and (
            job.qa_status == QAStatus.PASSED or not settings.qa_enabled
        ):
            logger.info("plan_job_already_done", video_job_id=video_job_id, stage=job.stage)
            scene_ids = [str(s.id) for s in job.scenes]
            return {
                "success": True,
                "video_job_id": video_job_id,
                "stage": job.stage,
                "scene_ids": scene_ids,
                "skipped": True,
            }

        # Check if QA has permanently failed
        if job.qa_status == QAStatus.FAILED_QA:
            logger.info("plan_job_qa_permanently_failed", video_job_id=video_job_id)
            return {
                "success": False,
                "video_job_id": video_job_id,
                "stage": "failed_qa",
                "error": job.last_qa_error,
            }

        # Update status
        job.status = "running"
        job.stage = "planning"
        job.celery_task_id = task_id
        if not job.started_at:
            job.started_at = start_time
        session.commit()

        try:
            # Build optimization context from historical performance
            optimization_context = None
            try:
                context_builder = OptimizationContextBuilder(session)
                opt_context = context_builder.build(job.project_id)
                if opt_context.sample_size > 0:
                    optimization_context = opt_context.format_for_prompt()
                    logger.info(
                        "optimization_context_loaded",
                        video_job_id=video_job_id,
                        sample_size=opt_context.sample_size,
                        winning_patterns=len(opt_context.winning_patterns),
                    )
            except Exception as e:
                # Don't fail planning if context building fails
                logger.warning(
                    "optimization_context_build_failed",
                    video_job_id=video_job_id,
                    error=str(e),
                )

            # Build story context if job has a linked story
            story_context = None
            target_duration_seconds = None
            if job.story_id and job.story:
                story_context = {
                    "narrative_style": job.story.narrative_style,
                    "topic": job.story.topic,
                }
                # Use story's estimated duration to calculate scene count and durations
                target_duration_seconds = job.story.estimated_duration_seconds
                logger.info(
                    "story_context_loaded",
                    video_job_id=video_job_id,
                    story_id=str(job.story_id),
                    narrative_style=job.story.narrative_style,
                    target_duration=target_duration_seconds,
                )

            # Run planner
            planner = PlannerService()
            plan = run_async(
                planner.plan(
                    job.idea,
                    job.style_preset,
                    optimization_context,
                    story_context,
                    target_duration_seconds,
                )
            )

            # Update job with plan
            job.title = plan.title
            job.description = plan.description
            job.plan_data = plan.raw_response

            # Clear existing scenes if regenerating
            if job.scenes:
                for scene in job.scenes:
                    session.delete(scene)
                session.flush()

            # Create scenes
            scene_ids = []
            _preset = get_preset(job.style_preset)  # May be used for validation in the future

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

            session.flush()

            # Run QA check if enabled
            if settings.qa_enabled:
                job.qa_attempts = (job.qa_attempts or 0) + 1
                qa_service = QAService()
                qa_result = run_async(qa_service.check_plan(job, session))

                # Save QA result
                qa_service.save_qa_result(
                    session=session,
                    video_job_id=job_uuid,
                    check_type=QACheckType.PLAN_QA,
                    stage=QAStage.POST_PLANNING,
                    attempt_number=job.qa_attempts,
                    result=qa_result,
                )

                # Record QA metrics
                record_qa_check(
                    session=session,
                    video_job_id=job_uuid,
                    stage="post_planning",
                    passed=qa_result.passed,
                    hook_clarity=qa_result.hook_clarity_score,
                    coherence=qa_result.coherence_score,
                )

                if not qa_result.passed:
                    job.last_qa_error = qa_result.feedback

                    if job.qa_attempts >= settings.qa_max_regeneration_attempts:
                        # Exhausted retries - permanent failure
                        job.qa_status = QAStatus.FAILED_QA
                        job.status = "failed"
                        job.stage = "failed_qa"
                        session.commit()

                        # Record failure metric
                        record_job_failed(session, job_uuid, "planning", "qa_failed")

                        # Send alert
                        run_async(
                            alert_qa_failure(
                                video_job_id=job_uuid,
                                qa_stage="post_planning",
                                feedback=qa_result.feedback,
                                attempt=job.qa_attempts,
                                max_attempts=settings.qa_max_regeneration_attempts,
                                scores={
                                    "hook_clarity": qa_result.hook_clarity_score,  # type: ignore[dict-item]
                                    "coherence": qa_result.coherence_score,  # type: ignore[dict-item]
                                },
                            )
                        )

                        logger.error(
                            "plan_job_qa_failed_permanently",
                            video_job_id=video_job_id,
                            attempts=job.qa_attempts,
                            feedback=qa_result.feedback,
                        )

                        return {
                            "success": False,
                            "video_job_id": video_job_id,
                            "stage": "failed_qa",
                            "qa_attempts": job.qa_attempts,
                            "error": qa_result.feedback,
                        }
                    else:
                        # Can retry - raise exception to trigger regeneration
                        job.qa_status = QAStatus.FAILED
                        session.commit()

                        logger.warning(
                            "plan_job_qa_failed_regenerating",
                            video_job_id=video_job_id,
                            attempt=job.qa_attempts,
                            max_attempts=settings.qa_max_regeneration_attempts,
                            feedback=qa_result.feedback,
                        )

                        raise QAFailedException(
                            f"QA failed (attempt {job.qa_attempts}/{settings.qa_max_regeneration_attempts}): {qa_result.feedback}",
                            qa_result,
                        )

                # QA passed
                job.qa_status = QAStatus.PASSED
            else:
                job.qa_status = QAStatus.SKIPPED

            job.stage = "planned"

            # Record success metrics
            duration = (datetime.now(UTC) - start_time).total_seconds()
            record_job_completed(session, job_uuid, "planning", duration)
            record_cost_estimate(session, job_uuid, estimate_planning_cost())

            session.commit()

            logger.info(
                "plan_job_completed",
                video_job_id=video_job_id,
                title=plan.title,
                scene_count=len(scene_ids),
                qa_status=job.qa_status,
            )

            return {
                "success": True,
                "video_job_id": video_job_id,
                "title": plan.title,
                "scene_ids": scene_ids,
                "total_duration": plan.total_duration,
                "qa_status": job.qa_status,
            }

        except QAFailedException:
            # Re-raise QA exceptions to trigger retry
            raise

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.retry_count = (job.retry_count or 0) + 1
            session.commit()

            # Record failure
            record_job_failed(session, job_uuid, "planning", type(e).__name__)

            # Send alert for pipeline failure
            run_async(
                alert_pipeline_failure(
                    video_job_id=job_uuid,
                    stage="planning",
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            )

            raise


# =============================================================================
# PIPELINE STAGE 2: Generate Scene Clips
# =============================================================================


def _generate_single_scene_clip(
    scene_id: str,
    video_job_id: str,
    previous_clip_path: str | None = None,
) -> dict[str, Any]:
    """Core scene clip generation logic.

    Used by both the Celery task and sequential frame chaining loop.

    Args:
        scene_id: UUID of the scene to generate.
        video_job_id: UUID of the parent video job.
        previous_clip_path: Optional path to previous clip for frame chaining.

    Returns:
        Dict with asset information.
    """
    scene_uuid = UUID(scene_id)
    job_uuid = UUID(video_job_id)

    logger.info(
        "generate_scene_clip_started",
        scene_id=scene_id,
        video_job_id=video_job_id,
        previous_clip_path=previous_clip_path,
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
                "file_path": existing_asset.file_path,
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

            # Frame chaining: extract last frame from previous clip
            reference_images = None
            if previous_clip_path and settings.video_frame_chaining_enabled:
                from shorts_engine.utils.frame_extraction import extract_last_frame

                clip_path = Path(previous_clip_path)
                if clip_path.exists():
                    frame_bytes = extract_last_frame(clip_path)
                    reference_images = [frame_bytes]
                    logger.info(
                        "frame_chaining_extracted",
                        scene_id=scene_id,
                        previous_clip=previous_clip_path,
                        frame_size=len(frame_bytes),
                    )
                else:
                    logger.warning(
                        "frame_chaining_clip_not_found",
                        scene_id=scene_id,
                        previous_clip=previous_clip_path,
                    )

            # Generate video
            provider = get_video_gen_provider()
            request = VideoGenRequest(
                prompt=full_prompt,
                duration_seconds=int(scene.duration_seconds),
                aspect_ratio="9:16",  # Vertical for shorts
                reference_images=reference_images,
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
                # Get download headers if provider requires authentication (e.g., Veo)
                download_headers = (
                    result.metadata.get("download_headers") if result.metadata else None
                )
                stored = run_async(
                    storage.store_from_url(
                        url=video_url,
                        asset_type="scene_clip",
                        video_job_id=job_uuid,
                        scene_id=scene_uuid,
                        metadata={
                            "provider": provider.name,
                            "generation_id": (
                                result.metadata.get("generation_id") if result.metadata else None
                            ),
                        },
                        headers=download_headers,
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
    name="pipeline.generate_scene_clip",
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
)
def generate_scene_clip_task(
    self: Any,
    scene_id: str,
    video_job_id: str,
) -> dict[str, Any]:
    """Generate a video clip for a single scene.

    Delegates to _generate_single_scene_clip for the actual work.

    Args:
        scene_id: UUID of the scene to generate
        video_job_id: UUID of the parent video job

    Returns:
        Dict with asset information
    """
    task_id = self.request.id
    logger.info("generate_scene_clip_task_dispatched", task_id=task_id, scene_id=scene_id)
    return _generate_single_scene_clip(scene_id, video_job_id)


@celery_app.task(
    bind=True,
    name="pipeline.generate_all_scenes",
    soft_time_limit=1800,  # 30 min soft limit (sequential frame chaining ~75s/scene)
    time_limit=1860,  # 31 min hard limit
)
def generate_all_scenes_task(
    _self: Any,  # noqa: ARG001
    video_job_id: str,
    scene_ids: list[str],
) -> dict[str, Any]:
    """Orchestrate generation of all scene video clips.

    When frame chaining is enabled, generates scenes sequentially so each
    scene can use the last frame of the previous clip as a reference image
    (image-to-video instead of text-to-video).

    When Ralph loop is enabled, delegates to ralph_loop_task
    for agentic retry with quality checks.

    Otherwise dispatches individual scene tasks with staggered delays.
    """
    ralph_enabled = settings.ralph_loop_enabled
    frame_chaining = settings.video_frame_chaining_enabled

    logger.info(
        "generate_all_scenes_started",
        video_job_id=video_job_id,
        scene_count=len(scene_ids),
        ralph_enabled=ralph_enabled,
        frame_chaining=frame_chaining,
    )

    with get_session_context() as session:
        job = session.get(VideoJobModel, UUID(video_job_id))
        if job:
            job.stage = "generating"
            if ralph_enabled:
                job.ralph_loop_enabled = True
                job.ralph_max_iterations = settings.ralph_max_iterations
            session.commit()

    # Use Ralph loop when enabled (takes priority over frame chaining)
    if ralph_enabled:
        from shorts_engine.jobs.ralph_tasks import ralph_loop_task

        logger.info(
            "generate_all_scenes_using_ralph",
            video_job_id=video_job_id,
            max_iterations=settings.ralph_max_iterations,
        )

        result = ralph_loop_task.apply_async(args=[video_job_id, scene_ids])
        try:
            ralph_result: dict[str, Any] = result.get(timeout=7200)  # 2 hours max for ralph loop
            return ralph_result
        except Exception as e:
            logger.error(
                "generate_all_scenes_ralph_failed",
                video_job_id=video_job_id,
                error=str(e),
            )
            return {
                "success": False,
                "video_job_id": video_job_id,
                "error": f"Ralph loop failed: {str(e)}",
            }

    # Frame chaining: generate scenes sequentially, passing last frame between clips
    if frame_chaining:
        logger.info(
            "generate_all_scenes_frame_chaining",
            video_job_id=video_job_id,
            scene_count=len(scene_ids),
        )

        previous_clip_path: str | None = None
        results = []

        for scene_id in scene_ids:
            try:
                result = _generate_single_scene_clip(
                    scene_id, video_job_id, previous_clip_path
                )
                results.append(result)

                # Chain: pass file_path of successful clip to next scene
                if result.get("success") and result.get("file_path"):
                    previous_clip_path = result["file_path"]
                else:
                    previous_clip_path = None  # Reset chain on failure
            except Exception as e:
                logger.error(
                    "generate_scene_clip_error_in_chain",
                    scene_id=scene_id,
                    video_job_id=video_job_id,
                    error=str(e),
                )
                results.append({
                    "success": False,
                    "scene_id": scene_id,
                    "error": str(e),
                })
                previous_clip_path = None  # Reset chain on failure

        success_count = sum(1 for r in results if r.get("success"))
        logger.info(
            "generate_all_scenes_completed",
            video_job_id=video_job_id,
            success_count=success_count,
            total_count=len(scene_ids),
            mode="frame_chaining",
        )

        return {
            "success": success_count == len(scene_ids),
            "video_job_id": video_job_id,
            "results": results,
            "success_count": success_count,
            "total_count": len(scene_ids),
        }

    # Standard generation (no Ralph loop, no frame chaining) - staggered to avoid rate limits
    rate_limit_seconds = settings.video_gen_rate_limit_seconds

    logger.info(
        "generate_all_scenes_staggered_dispatch",
        video_job_id=video_job_id,
        scene_count=len(scene_ids),
        rate_limit_seconds=rate_limit_seconds,
    )

    # Dispatch tasks with staggered countdown delays
    async_results = []
    for i, scene_id in enumerate(scene_ids):
        countdown = i * rate_limit_seconds  # 0, 6, 12, 18, ... seconds
        result = generate_scene_clip_task.apply_async(
            args=[scene_id, video_job_id],
            countdown=countdown,
        )
        async_results.append(result)

    # Wait for all tasks to complete (with timeout)
    try:
        results = []
        # Total timeout: base 1 hour + stagger time for all scenes
        total_timeout = 3600 + (len(scene_ids) * rate_limit_seconds)

        for async_result in async_results:
            result = async_result.get(timeout=total_timeout)
            results.append(result)

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
    self: Any,
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
        scenes = (
            session.execute(
                select(SceneModel)
                .where(SceneModel.video_job_id == job_uuid)
                .order_by(SceneModel.scene_number)
            )
            .scalars()
            .all()
        )

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
                failed_assets.append(
                    {
                        "scene_id": str(scene.id),
                        "scene_number": scene.scene_number,
                        "error": "No asset found",
                    }
                )
                continue

            if asset.status != "ready":
                failed_assets.append(
                    {
                        "scene_id": str(scene.id),
                        "scene_number": scene.scene_number,
                        "error": f"Asset status: {asset.status}",
                    }
                )
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
                verified_assets.append(
                    {
                        "scene_id": str(scene.id),
                        "scene_number": scene.scene_number,
                        "asset_id": str(asset.id),
                        "url": asset.url or f"file://{asset.file_path}",
                    }
                )
            else:
                failed_assets.append(
                    {
                        "scene_id": str(scene.id),
                        "scene_number": scene.scene_number,
                        "error": "Asset verification failed",
                    }
                )

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
    self: Any,  # noqa: ARG001
    video_job_id_or_result: str | dict[str, Any],
    verification_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark a video job as ready for rendering.

    When called from a Celery chain, the first arg may be the verification
    result dict (from verify_assets_task) rather than a video_job_id string.

    Args:
        video_job_id_or_result: Either a video job UUID string or the
            verification result dict (which contains video_job_id).
        verification_result: Optional result from verify_assets (used when
            called directly with video_job_id as first arg).

    Returns:
        Dict with final status
    """
    task_id = self.request.id

    # Handle chain: verify_assets returns a dict as first positional arg
    if isinstance(video_job_id_or_result, dict):
        verification_result = video_job_id_or_result
        video_job_id = video_job_id_or_result["video_job_id"]
    else:
        video_job_id = video_job_id_or_result

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
            job.error_message = (
                f"Asset verification failed: {verification_result.get('failed_assets')}"
            )
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
        job.completed_at = datetime.now(UTC)
        session.commit()

        # Collect final asset URLs
        assets = (
            session.execute(
                select(AssetModel).where(
                    AssetModel.video_job_id == job_uuid,
                    AssetModel.asset_type == "scene_clip",
                    AssetModel.status == "ready",
                )
            )
            .scalars()
            .all()
        )

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
    self: Any,  # noqa: ARG001
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
            session.flush()  # Get the job ID before story generation

            # Generate a story from the idea so the render pipeline has
            # a proper narration script instead of falling back to caption beats.
            try:
                from shorts_engine.services.story_generator import StoryGenerator

                generator = StoryGenerator()
                story = run_async(generator.generate(idea))

                story_model = StoryModel(
                    id=uuid4(),
                    project_id=project_uuid,
                    topic=idea,
                    title=story.title,
                    narrative_text=story.narrative_text,
                    narrative_style=story.narrative_style,
                    suggested_preset=story.suggested_preset,
                    word_count=story.word_count,
                    estimated_duration_seconds=story.estimated_duration_seconds,
                    status="approved",
                )
                session.add(story_model)
                session.flush()

                job.story_id = story_model.id
                job.idea = story.narrative_text  # Full narrative replaces short idea
                if not style_preset:
                    job.style_preset = story.suggested_preset

                logger.info(
                    "full_pipeline_story_generated",
                    video_job_id=str(job.id),
                    story_id=str(story_model.id),
                    word_count=story.word_count,
                )
            except Exception:
                logger.warning(
                    "full_pipeline_story_generation_failed",
                    video_job_id=str(job.id),
                    exc_info=True,
                )
                # Continue without a story — pipeline falls back to caption beats

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
    soft_time_limit=1800,
    time_limit=1860,
)
def generate_all_scenes_from_plan(
    _self: Any,  # noqa: ARG001
    plan_result: dict[str, Any],
    video_job_id: str,
) -> str:
    """Extract scene IDs from plan result and trigger generation.

    This is a bridge task between planning and generation stages.
    Calls generate_all_scenes_task inline to avoid the Celery
    result.get()-within-a-task bug.
    """
    if not plan_result.get("success"):
        raise RuntimeError(f"Planning failed: {plan_result.get('error', 'Unknown error')}")

    scene_ids = plan_result.get("scene_ids", [])
    if not scene_ids:
        raise RuntimeError("No scenes in plan result")

    # Call generation inline (not as sub-task) to avoid result.get() deadlock
    gen_result = generate_all_scenes_task(video_job_id, scene_ids)

    if not gen_result.get("success"):
        raise RuntimeError(f"Scene generation failed: {gen_result.get('error', 'Unknown error')}")

    # Return video_job_id for next stage
    return video_job_id
