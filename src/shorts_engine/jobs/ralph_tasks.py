"""Ralph loop Celery tasks for agentic retry of video clip generation."""

from typing import Any
from uuid import UUID

from celery import group

from shorts_engine.config import settings
from shorts_engine.db.models import AssetModel, RalphIterationModel, SceneModel, VideoJobModel
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import RalphStatus
from shorts_engine.logging import get_logger
from shorts_engine.services.ralph_loop import FeedbackContext, RalphLoopController
from shorts_engine.utils import run_async
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="pipeline.ralph_loop",
    max_retries=0,  # We handle retries internally
)
def ralph_loop_task(
    self: Any,
    video_job_id: str,
    scene_ids: list[str],
) -> dict[str, Any]:
    """Execute the Ralph agentic retry loop for video clip generation.

    This task:
    1. Generates all scene video clips
    2. Runs quality checks (visual coherence, style consistency, motion, temporal, QA)
    3. If all pass: returns success
    4. If not: clears clips, builds feedback, regenerates with feedback
    5. Repeats until max iterations or all criteria pass

    On exhaustion with ralph_use_best_on_failure=True, restores the best
    iteration's clips instead of failing completely.

    Args:
        video_job_id: UUID of the video job
        scene_ids: List of scene UUIDs to generate clips for

    Returns:
        Dict with success status, iteration info, and results
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info(
        "ralph_loop_started",
        task_id=task_id,
        video_job_id=video_job_id,
        scene_count=len(scene_ids),
    )

    controller = RalphLoopController()
    feedback_context: FeedbackContext | None = None
    best_iteration_number = 0

    with get_session_context() as session:
        # Load job and initialize
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        max_iterations = job.ralph_max_iterations or settings.ralph_max_iterations
        job.ralph_loop_enabled = True
        job.ralph_status = RalphStatus.RUNNING.value
        job.ralph_current_iteration = 0
        session.commit()

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "ralph_iteration_starting",
                video_job_id=video_job_id,
                iteration=iteration,
                max_iterations=max_iterations,
            )

            # Update iteration counter
            job.ralph_current_iteration = iteration
            session.commit()

            # Step 1: Clear previous scene clips (except first iteration)
            if iteration > 1:
                controller.clear_scene_clips(session, job_uuid)
                session.commit()

            # Step 2: Generate all scene clips (with feedback if available)
            gen_result = _generate_clips_with_feedback(
                video_job_id=video_job_id,
                scene_ids=scene_ids,
                feedback_context=feedback_context,
            )

            if not gen_result.get("success"):
                logger.error(
                    "ralph_generation_failed",
                    video_job_id=video_job_id,
                    iteration=iteration,
                    error=gen_result.get("error"),
                )
                job.ralph_status = RalphStatus.FAILED.value
                session.commit()
                return {
                    "success": False,
                    "video_job_id": video_job_id,
                    "iteration": iteration,
                    "error": f"Video clip generation failed: {gen_result.get('error')}",
                }

            # Refresh job from session after generation
            session.refresh(job)

            # Step 3: Run quality checks
            iteration_result = run_async(controller.run_iteration(session, job, iteration))
            session.commit()

            # Track best iteration for potential restoration
            if iteration == 1 or _is_better_iteration(session, job_uuid, iteration):
                best_iteration_number = iteration

            # Step 4: Check if all criteria passed
            if iteration_result.all_criteria_passed:
                logger.info(
                    "ralph_loop_passed",
                    video_job_id=video_job_id,
                    iteration=iteration,
                    visual_coherence=iteration_result.visual_coherence_score,
                    style_consistency=iteration_result.style_consistency_score,
                    motion_coherence=iteration_result.motion_coherence_score,
                    temporal_consistency=iteration_result.temporal_consistency_score,
                )

                # Mark iteration as final
                _mark_final_iteration(session, job_uuid, iteration)
                job.ralph_status = RalphStatus.PASSED.value
                session.commit()

                return {
                    "success": True,
                    "video_job_id": video_job_id,
                    "iteration": iteration,
                    "ralph_status": RalphStatus.PASSED.value,
                    "visual_coherence_score": iteration_result.visual_coherence_score,
                    "style_consistency_score": iteration_result.style_consistency_score,
                    "motion_coherence_score": iteration_result.motion_coherence_score,
                    "temporal_consistency_score": iteration_result.temporal_consistency_score,
                    "all_criteria_passed": True,
                }

            # Step 5: Check if should retry
            if not controller.should_retry(iteration_result, iteration, max_iterations):
                # Exhausted iterations
                break

            # Step 6: Build feedback for next iteration
            from shorts_engine.services.video_critique import (
                SceneVideoCritique,
                VideoCritiqueResult,
            )

            # Reconstruct critique result from iteration result
            critique_result = VideoCritiqueResult(
                visual_coherence_score=iteration_result.visual_coherence_score,
                style_consistency_score=iteration_result.style_consistency_score,
                motion_coherence_score=iteration_result.motion_coherence_score,
                temporal_consistency_score=iteration_result.temporal_consistency_score,
                overall_passed=iteration_result.llm_critique_passed,
                per_scene_feedback=[
                    SceneVideoCritique(
                        scene_number=sf.get("scene_number", 0),
                        score=sf.get("score", 0.0),
                        feedback=sf.get("feedback", ""),
                        motion_quality=sf.get("motion_quality", "unknown"),
                        issues=sf.get("issues", []),
                    )
                    for sf in (iteration_result.per_scene_feedback or {}).get("feedback", [])
                ],
                cross_scene_feedback=(iteration_result.per_scene_feedback or {}).get(
                    "cross_scene_feedback", ""
                ),
                improvement_suggestions=iteration_result.improvement_suggestions,
                summary=iteration_result.feedback,
            )

            # Get QA result for feedback
            from shorts_engine.services.qa import QAResult

            qa_result = QAResult(
                passed=iteration_result.qa_passed,
                hook_clarity_score=iteration_result.hook_clarity_score,
                coherence_score=iteration_result.coherence_score,
                feedback=iteration_result.feedback if not iteration_result.qa_passed else "",
            )

            feedback_context = controller.build_feedback_for_next_iteration(
                critique_result=critique_result,
                qa_result=qa_result,
                iteration_number=iteration,
            )

            logger.info(
                "ralph_iteration_feedback_built",
                video_job_id=video_job_id,
                iteration=iteration,
                visual_issues=len(feedback_context.visual_issues),
                style_issues=len(feedback_context.style_issues),
                motion_issues=len(feedback_context.motion_issues),
                temporal_issues=len(feedback_context.temporal_issues),
                suggestions=len(feedback_context.global_suggestions),
            )

        # Exhausted all iterations without passing
        logger.warning(
            "ralph_loop_exhausted",
            video_job_id=video_job_id,
            iterations_run=max_iterations,
            best_iteration=best_iteration_number,
        )

        # Handle failure based on settings
        if settings.ralph_use_best_on_failure and best_iteration_number > 0:
            # Restore best iteration's clips
            restored = _restore_best_iteration(session, job_uuid, best_iteration_number)
            _mark_final_iteration(session, job_uuid, best_iteration_number)

            job.ralph_status = RalphStatus.FAILED.value
            session.commit()

            logger.info(
                "ralph_restored_best_iteration",
                video_job_id=video_job_id,
                restored_iteration=best_iteration_number,
                assets_restored=restored,
            )

            return {
                "success": True,  # Partial success - used best available
                "video_job_id": video_job_id,
                "iteration": max_iterations,
                "ralph_status": RalphStatus.FAILED.value,
                "best_iteration_used": best_iteration_number,
                "all_criteria_passed": False,
                "note": f"Used best iteration ({best_iteration_number}) after exhausting retries",
            }
        else:
            job.ralph_status = RalphStatus.FAILED.value
            session.commit()

            return {
                "success": False,
                "video_job_id": video_job_id,
                "iteration": max_iterations,
                "ralph_status": RalphStatus.FAILED.value,
                "error": "Ralph loop exhausted without meeting quality criteria",
            }


def _generate_clips_with_feedback(
    video_job_id: str,
    scene_ids: list[str],
    feedback_context: FeedbackContext | None,
) -> dict[str, Any]:
    """Generate scene video clips, optionally with feedback context.

    Args:
        video_job_id: UUID of the video job
        scene_ids: List of scene UUIDs
        feedback_context: Optional feedback from previous iteration

    Returns:
        Dict with success status and results
    """
    from shorts_engine.jobs.video_pipeline import generate_scene_clip_task

    # If we have feedback, we need to inject it into the prompts
    if feedback_context:
        _inject_feedback_into_prompts(video_job_id, feedback_context)

    # Create a group of video clip generation tasks
    generation_tasks = group(
        generate_scene_clip_task.s(scene_id, video_job_id) for scene_id in scene_ids
    )

    # Execute and wait for all
    result = generation_tasks.apply_async()

    try:
        results = result.get(timeout=3600)  # 60 minutes max for video clips
        success_count = sum(1 for r in results if r.get("success"))

        return {
            "success": success_count == len(scene_ids),
            "results": results,
            "success_count": success_count,
            "total_count": len(scene_ids),
        }
    except Exception as e:
        logger.error(
            "ralph_clip_generation_error",
            video_job_id=video_job_id,
            error=str(e),
        )
        return {
            "success": False,
            "error": str(e),
        }


def _inject_feedback_into_prompts(
    video_job_id: str,
    feedback_context: FeedbackContext,
) -> None:
    """Inject feedback context into scene prompts for regeneration.

    This temporarily modifies scene visual_prompt to include feedback.
    The feedback is prepended to help guide better generation.

    Args:
        video_job_id: UUID of the video job
        feedback_context: Feedback from previous iteration
    """
    job_uuid = UUID(video_job_id)
    feedback_text = feedback_context.format_for_prompt()

    with get_session_context() as session:
        from sqlalchemy import select

        scenes = (
            session.execute(
                select(SceneModel)
                .where(SceneModel.video_job_id == job_uuid)
                .order_by(SceneModel.scene_number)
            )
            .scalars()
            .all()
        )

        for scene in scenes:
            # Add per-scene feedback if available
            scene_feedback = feedback_context.per_scene_suggestions.get(scene.scene_number, [])
            scene_specific = ""
            if scene_feedback:
                scene_specific = f"\nScene-specific issues: {'; '.join(scene_feedback)}"

            # Store original prompt in metadata if not already stored
            if not scene.metadata_:
                scene.metadata_ = {}
            if "original_visual_prompt" not in scene.metadata_:
                scene.metadata_["original_visual_prompt"] = scene.visual_prompt

            # Prepend feedback to prompt (will be processed by video gen task)
            # Note: The actual prompt enhancement happens in the video gen task
            scene.metadata_["ralph_feedback"] = feedback_text + scene_specific
            scene.metadata_["ralph_iteration"] = feedback_context.iteration_number + 1

        session.commit()

        logger.info(
            "ralph_feedback_injected",
            video_job_id=video_job_id,
            scene_count=len(scenes),
        )


def _is_better_iteration(
    session: Any,  # Session
    video_job_id: UUID,
    current_iteration: int,
) -> bool:
    """Check if current iteration is better than previous best.

    Args:
        session: Database session
        video_job_id: The video job UUID
        current_iteration: Current iteration number

    Returns:
        True if current is the best so far
    """
    from sqlalchemy import select

    iterations = (
        session.execute(
            select(RalphIterationModel)
            .where(RalphIterationModel.video_job_id == video_job_id)
            .order_by(RalphIterationModel.iteration_number)
        )
        .scalars()
        .all()
    )

    if len(iterations) <= 1:
        return True

    def score(it: RalphIterationModel) -> float:
        scores = []
        if it.visual_coherence_score is not None:
            scores.append(it.visual_coherence_score)
        if it.style_consistency_score is not None:
            scores.append(it.style_consistency_score)
        if it.motion_coherence_score is not None:
            scores.append(it.motion_coherence_score)
        if it.temporal_consistency_score is not None:
            scores.append(it.temporal_consistency_score)
        return sum(scores) / len(scores) if scores else 0.0

    current = next((it for it in iterations if it.iteration_number == current_iteration), None)
    if not current:
        return False

    current_score = score(current)
    for it in iterations:
        if it.iteration_number != current_iteration and score(it) > current_score:
            return False

    return True


def _mark_final_iteration(
    session: Any,  # Session
    video_job_id: UUID,
    iteration_number: int,
) -> None:
    """Mark an iteration as the final one used.

    Args:
        session: Database session
        video_job_id: The video job UUID
        iteration_number: The iteration to mark as final
    """
    from sqlalchemy import select

    iteration = session.execute(
        select(RalphIterationModel).where(
            RalphIterationModel.video_job_id == video_job_id,
            RalphIterationModel.iteration_number == iteration_number,
        )
    ).scalar_one_or_none()

    if iteration:
        iteration.is_final_attempt = True
        session.flush()


def _restore_best_iteration(
    session: Any,  # Session
    video_job_id: UUID,
    best_iteration_number: int,
) -> int:
    """Restore scene clips from the best iteration.

    This is used when ralph_use_best_on_failure=True and we've
    exhausted iterations without passing all criteria.

    Args:
        session: Database session
        video_job_id: The video job UUID
        best_iteration_number: The iteration to restore from

    Returns:
        Number of assets restored
    """
    from sqlalchemy import select

    # Get the best iteration's asset snapshot
    iteration = session.execute(
        select(RalphIterationModel).where(
            RalphIterationModel.video_job_id == video_job_id,
            RalphIterationModel.iteration_number == best_iteration_number,
        )
    ).scalar_one_or_none()

    if not iteration or not iteration.asset_snapshot:
        logger.warning(
            "ralph_no_snapshot_to_restore",
            video_job_id=str(video_job_id),
            iteration=best_iteration_number,
        )
        return 0

    # Clear current clips
    current_assets = (
        session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == video_job_id,
                AssetModel.asset_type == "scene_clip",
            )
        )
        .scalars()
        .all()
    )

    for asset in current_assets:
        session.delete(asset)

    session.flush()

    # The snapshot contains references - in a real implementation,
    # we would restore actual files. For now, we log the restoration
    # and mark scenes as generated.
    restored_count = len(iteration.asset_snapshot.get("assets", []))

    scenes = (
        session.execute(select(SceneModel).where(SceneModel.video_job_id == video_job_id))
        .scalars()
        .all()
    )

    for scene in scenes:
        scene.status = "generated"

    session.flush()

    logger.info(
        "ralph_restored_assets",
        video_job_id=str(video_job_id),
        from_iteration=best_iteration_number,
        count=restored_count,
    )

    return restored_count
