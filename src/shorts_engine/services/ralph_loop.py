"""Ralph loop controller for agentic retry of video clip generation."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shorts_engine.config import settings
from shorts_engine.db.models import AssetModel, RalphIterationModel, SceneModel, VideoJobModel
from shorts_engine.domain.enums import RalphStatus
from shorts_engine.logging import get_logger
from shorts_engine.services.qa import QAResult, QAService
from shorts_engine.services.video_critique import VideoCritiqueResult, VideoCritiqueService

logger = get_logger(__name__)


@dataclass
class IterationResult:
    """Result of a single Ralph loop iteration."""

    iteration_number: int
    visual_coherence_score: float
    style_consistency_score: float
    motion_coherence_score: float
    temporal_consistency_score: float
    hook_clarity_score: float | None
    coherence_score: float | None
    visual_coherence_passed: bool
    style_consistency_passed: bool
    motion_coherence_passed: bool
    temporal_consistency_passed: bool
    qa_passed: bool
    llm_critique_passed: bool
    all_criteria_passed: bool
    feedback: str
    improvement_suggestions: list[str] = field(default_factory=list)
    per_scene_feedback: dict[str, Any] | None = None


@dataclass
class FeedbackContext:
    """Aggregated feedback to guide next iteration's generation."""

    iteration_number: int
    visual_issues: list[str]
    style_issues: list[str]
    motion_issues: list[str]
    temporal_issues: list[str]
    per_scene_suggestions: dict[int, list[str]]
    global_suggestions: list[str]

    def format_for_prompt(self) -> str:
        """Format feedback as additional prompt context for video generation."""
        parts = [
            "IMPORTANT: The previous iteration had quality issues. Please address:",
        ]

        if self.visual_issues:
            parts.append("\nVisual coherence issues:")
            for issue in self.visual_issues:
                parts.append(f"  - {issue}")

        if self.style_issues:
            parts.append("\nStyle consistency issues:")
            for issue in self.style_issues:
                parts.append(f"  - {issue}")

        if self.motion_issues:
            parts.append("\nMotion coherence issues:")
            for issue in self.motion_issues:
                parts.append(f"  - {issue}")

        if self.temporal_issues:
            parts.append("\nTemporal consistency issues:")
            for issue in self.temporal_issues:
                parts.append(f"  - {issue}")

        if self.global_suggestions:
            parts.append("\nSuggestions for improvement:")
            for suggestion in self.global_suggestions:
                parts.append(f"  - {suggestion}")

        return "\n".join(parts)


class RalphLoopController:
    """Controller for the Ralph agentic retry loop.

    The Ralph loop:
    1. Generates all scene video clips
    2. Runs quality checks (visual coherence, style consistency, motion, temporal, QA)
    3. If all pass: done
    4. If not: build feedback and regenerate
    5. Repeat until max iterations or all pass
    """

    def __init__(
        self,
        critique_service: VideoCritiqueService | None = None,
        qa_service: QAService | None = None,
    ) -> None:
        """Initialize the Ralph loop controller.

        Args:
            critique_service: Optional video critique service
            qa_service: Optional QA service for hook/coherence checks
        """
        self.critique_service = critique_service or VideoCritiqueService()
        self.qa_service = qa_service or QAService()

    def should_retry(
        self,
        iteration_result: IterationResult,
        current_iteration: int,
        max_iterations: int,
    ) -> bool:
        """Determine if another iteration should be attempted.

        Args:
            iteration_result: Result from the current iteration
            current_iteration: Current iteration number (1-indexed)
            max_iterations: Maximum allowed iterations

        Returns:
            True if should retry, False if done (passed or exhausted)
        """
        # Don't retry if all criteria passed
        if iteration_result.all_criteria_passed:
            logger.info(
                "ralph_no_retry_passed",
                iteration=current_iteration,
            )
            return False

        # Don't retry if at max iterations
        if current_iteration >= max_iterations:
            logger.info(
                "ralph_no_retry_exhausted",
                iteration=current_iteration,
                max_iterations=max_iterations,
            )
            return False

        # Should retry
        logger.info(
            "ralph_should_retry",
            iteration=current_iteration,
            max_iterations=max_iterations,
            visual_coherence_passed=iteration_result.visual_coherence_passed,
            style_consistency_passed=iteration_result.style_consistency_passed,
            motion_coherence_passed=iteration_result.motion_coherence_passed,
            temporal_consistency_passed=iteration_result.temporal_consistency_passed,
        )
        return True

    def build_feedback_for_next_iteration(
        self,
        critique_result: VideoCritiqueResult,
        qa_result: QAResult | None,
        iteration_number: int,
    ) -> FeedbackContext:
        """Build consolidated feedback to guide the next iteration.

        Args:
            critique_result: Video critique result from current iteration
            qa_result: Optional QA result from current iteration
            iteration_number: The iteration that produced these results

        Returns:
            FeedbackContext with aggregated suggestions
        """
        visual_issues: list[str] = []
        style_issues: list[str] = []
        motion_issues: list[str] = []
        temporal_issues: list[str] = []
        per_scene: dict[int, list[str]] = {}
        global_suggestions: list[str] = []

        # Extract issues from critique summary
        if critique_result.visual_coherence_score < settings.ralph_visual_coherence_threshold:
            visual_issues.append(
                f"Visual coherence too low ({critique_result.visual_coherence_score:.2f})"
            )

        if critique_result.style_consistency_score < settings.ralph_style_consistency_threshold:
            style_issues.append(
                f"Style consistency too low ({critique_result.style_consistency_score:.2f})"
            )

        if critique_result.motion_coherence_score < settings.ralph_motion_coherence_threshold:
            motion_issues.append(
                f"Motion coherence too low ({critique_result.motion_coherence_score:.2f})"
            )

        if (
            critique_result.temporal_consistency_score
            < settings.ralph_temporal_consistency_threshold
        ):
            temporal_issues.append(
                f"Temporal consistency too low ({critique_result.temporal_consistency_score:.2f})"
            )

        # Per-scene feedback
        for scene_fb in critique_result.per_scene_feedback:
            if scene_fb.issues:
                per_scene[scene_fb.scene_number] = scene_fb.issues
            # Add motion-specific feedback
            if scene_fb.motion_quality in ("jerky", "static"):
                per_scene.setdefault(scene_fb.scene_number, []).append(
                    f"Motion quality: {scene_fb.motion_quality}"
                )

        # Add cross-scene feedback as global suggestion
        if critique_result.cross_scene_feedback:
            global_suggestions.append(f"Scene transitions: {critique_result.cross_scene_feedback}")

        # Global suggestions
        global_suggestions.extend(critique_result.improvement_suggestions)

        # Add QA feedback if available
        if qa_result and not qa_result.passed and qa_result.feedback:
            global_suggestions.append(f"QA feedback: {qa_result.feedback}")

        return FeedbackContext(
            iteration_number=iteration_number,
            visual_issues=visual_issues,
            style_issues=style_issues,
            motion_issues=motion_issues,
            temporal_issues=temporal_issues,
            per_scene_suggestions=per_scene,
            global_suggestions=global_suggestions,
        )

    async def run_iteration(
        self,
        session: Session,
        video_job: VideoJobModel,
        iteration_number: int,
    ) -> IterationResult:
        """Execute quality checks for a single iteration.

        This assumes video clips have already been generated.

        Args:
            session: Database session
            video_job: The video job being processed
            iteration_number: Current iteration number

        Returns:
            IterationResult with all scores and pass/fail status
        """
        job_uuid = video_job.id
        logger.info(
            "ralph_iteration_started",
            video_job_id=str(job_uuid),
            iteration=iteration_number,
        )

        # Create iteration record
        iteration = RalphIterationModel(
            video_job_id=job_uuid,
            iteration_number=iteration_number,
            started_at=datetime.now(UTC),
        )
        session.add(iteration)
        session.flush()

        # Get scene video clips
        scenes = (
            session.execute(
                select(SceneModel)
                .where(SceneModel.video_job_id == job_uuid)
                .order_by(SceneModel.scene_number)
            )
            .scalars()
            .all()
        )

        clip_urls: list[str] = []
        scene_ids: list[UUID] = []
        scene_descriptions: list[str] = []
        asset_snapshot: dict[str, Any] = {"assets": []}

        for scene in scenes:
            # Get the scene clip asset
            asset = session.execute(
                select(AssetModel).where(
                    AssetModel.scene_id == scene.id,
                    AssetModel.asset_type == "scene_clip",
                    AssetModel.status == "ready",
                )
            ).scalar_one_or_none()

            if asset:
                url = asset.url or f"file://{asset.file_path}"
                clip_urls.append(url)
                scene_ids.append(scene.id)
                scene_descriptions.append(scene.visual_prompt)

                # Snapshot for potential restoration
                asset_snapshot["assets"].append(
                    {
                        "scene_id": str(scene.id),
                        "asset_id": str(asset.id),
                        "url": url,
                        "file_path": asset.file_path,
                    }
                )

        # Store asset snapshot
        iteration.asset_snapshot = asset_snapshot

        # Run video critique
        critique_result = await self.critique_service.critique_scene_clips(
            clip_urls=clip_urls,
            scene_ids=scene_ids,
            style_preset=video_job.style_preset,
            scene_descriptions=scene_descriptions,
        )

        # Run QA check (hook clarity, coherence)
        qa_result = await self.qa_service.check_plan(video_job, session)

        # Calculate pass/fail for each criterion
        visual_coherence_passed = (
            critique_result.visual_coherence_score >= settings.ralph_visual_coherence_threshold
        )
        style_consistency_passed = (
            critique_result.style_consistency_score >= settings.ralph_style_consistency_threshold
        )
        motion_coherence_passed = (
            critique_result.motion_coherence_score >= settings.ralph_motion_coherence_threshold
        )
        temporal_consistency_passed = (
            critique_result.temporal_consistency_score
            >= settings.ralph_temporal_consistency_threshold
        )
        qa_passed = qa_result.passed
        llm_critique_passed = critique_result.overall_passed

        all_passed = (
            visual_coherence_passed
            and style_consistency_passed
            and motion_coherence_passed
            and temporal_consistency_passed
            and qa_passed
            and llm_critique_passed
        )

        # Update iteration record
        iteration.visual_coherence_score = critique_result.visual_coherence_score
        iteration.style_consistency_score = critique_result.style_consistency_score
        iteration.motion_coherence_score = critique_result.motion_coherence_score
        iteration.temporal_consistency_score = critique_result.temporal_consistency_score
        iteration.hook_clarity_score = qa_result.hook_clarity_score
        iteration.coherence_score = qa_result.coherence_score
        iteration.visual_coherence_passed = visual_coherence_passed
        iteration.style_consistency_passed = style_consistency_passed
        iteration.motion_coherence_passed = motion_coherence_passed
        iteration.temporal_consistency_passed = temporal_consistency_passed
        iteration.qa_passed = qa_passed
        iteration.llm_critique_passed = llm_critique_passed
        iteration.all_criteria_passed = all_passed
        iteration.llm_critique_feedback = critique_result.summary
        iteration.llm_critique_raw = critique_result.raw_response
        iteration.per_scene_feedback = {
            "feedback": [
                {
                    "scene_number": sf.scene_number,
                    "score": sf.score,
                    "feedback": sf.feedback,
                    "motion_quality": sf.motion_quality,
                    "issues": sf.issues,
                }
                for sf in critique_result.per_scene_feedback
            ],
            "cross_scene_feedback": critique_result.cross_scene_feedback,
        }
        iteration.completed_at = datetime.now(UTC)

        session.flush()

        # Build feedback string
        feedback_parts = [critique_result.summary]
        if not visual_coherence_passed:
            feedback_parts.append(
                f"Visual coherence: {critique_result.visual_coherence_score:.2f} "
                f"(threshold: {settings.ralph_visual_coherence_threshold})"
            )
        if not style_consistency_passed:
            feedback_parts.append(
                f"Style consistency: {critique_result.style_consistency_score:.2f} "
                f"(threshold: {settings.ralph_style_consistency_threshold})"
            )
        if not motion_coherence_passed:
            feedback_parts.append(
                f"Motion coherence: {critique_result.motion_coherence_score:.2f} "
                f"(threshold: {settings.ralph_motion_coherence_threshold})"
            )
        if not temporal_consistency_passed:
            feedback_parts.append(
                f"Temporal consistency: {critique_result.temporal_consistency_score:.2f} "
                f"(threshold: {settings.ralph_temporal_consistency_threshold})"
            )
        if not qa_passed:
            feedback_parts.append(f"QA: {qa_result.feedback}")

        logger.info(
            "ralph_iteration_completed",
            video_job_id=str(job_uuid),
            iteration=iteration_number,
            all_passed=all_passed,
            visual_coherence=critique_result.visual_coherence_score,
            style_consistency=critique_result.style_consistency_score,
            motion_coherence=critique_result.motion_coherence_score,
            temporal_consistency=critique_result.temporal_consistency_score,
        )

        return IterationResult(
            iteration_number=iteration_number,
            visual_coherence_score=critique_result.visual_coherence_score,
            style_consistency_score=critique_result.style_consistency_score,
            motion_coherence_score=critique_result.motion_coherence_score,
            temporal_consistency_score=critique_result.temporal_consistency_score,
            hook_clarity_score=qa_result.hook_clarity_score,
            coherence_score=qa_result.coherence_score,
            visual_coherence_passed=visual_coherence_passed,
            style_consistency_passed=style_consistency_passed,
            motion_coherence_passed=motion_coherence_passed,
            temporal_consistency_passed=temporal_consistency_passed,
            qa_passed=qa_passed,
            llm_critique_passed=llm_critique_passed,
            all_criteria_passed=all_passed,
            feedback=" | ".join(feedback_parts),
            improvement_suggestions=critique_result.improvement_suggestions,
            per_scene_feedback=iteration.per_scene_feedback,
        )

    def get_best_iteration(
        self,
        session: Session,
        video_job_id: UUID,
    ) -> RalphIterationModel | None:
        """Find the best iteration based on combined scores.

        Args:
            session: Database session
            video_job_id: The video job to search

        Returns:
            The iteration with highest combined score, or None
        """
        iterations = (
            session.execute(
                select(RalphIterationModel)
                .where(RalphIterationModel.video_job_id == video_job_id)
                .order_by(RalphIterationModel.iteration_number)
            )
            .scalars()
            .all()
        )

        if not iterations:
            return None

        # Score each iteration (average of available scores)
        def score_iteration(it: RalphIterationModel) -> float:
            scores = []
            if it.visual_coherence_score is not None:
                scores.append(it.visual_coherence_score)
            if it.style_consistency_score is not None:
                scores.append(it.style_consistency_score)
            if it.motion_coherence_score is not None:
                scores.append(it.motion_coherence_score)
            if it.temporal_consistency_score is not None:
                scores.append(it.temporal_consistency_score)
            if it.hook_clarity_score is not None:
                scores.append(it.hook_clarity_score)
            if it.coherence_score is not None:
                scores.append(it.coherence_score)
            return sum(scores) / len(scores) if scores else 0.0

        return max(iterations, key=score_iteration)

    def clear_scene_clips(
        self,
        session: Session,
        video_job_id: UUID,
    ) -> int:
        """Clear all scene clips for a video job before regeneration.

        Args:
            session: Database session
            video_job_id: The video job to clear clips for

        Returns:
            Number of assets deleted
        """
        # Get all scene_clip assets for this job
        assets = (
            session.execute(
                select(AssetModel).where(
                    AssetModel.video_job_id == video_job_id,
                    AssetModel.asset_type == "scene_clip",
                )
            )
            .scalars()
            .all()
        )

        count = len(assets)
        for asset in assets:
            session.delete(asset)

        # Reset scene status
        scenes = (
            session.execute(select(SceneModel).where(SceneModel.video_job_id == video_job_id))
            .scalars()
            .all()
        )
        for scene in scenes:
            scene.status = "pending"

        session.flush()

        logger.info(
            "ralph_cleared_clips",
            video_job_id=str(video_job_id),
            assets_deleted=count,
        )

        return count

    def update_job_ralph_status(
        self,
        session: Session,
        video_job: VideoJobModel,
        status: RalphStatus,
        current_iteration: int,
    ) -> None:
        """Update the Ralph loop status on the video job.

        Args:
            session: Database session
            video_job: The video job to update
            status: New Ralph status
            current_iteration: Current iteration number
        """
        video_job.ralph_status = status.value
        video_job.ralph_current_iteration = current_iteration
        session.flush()


def get_ralph_loop_controller() -> RalphLoopController:
    """Get a Ralph loop controller instance."""
    return RalphLoopController()
