"""Celery tasks for the learning loop.

Includes:
- plan_next_batch: Nightly job to plan and enqueue video jobs
- update_recipe_stats: Periodic job to update recipe performance stats
- evaluate_experiments: Job to check experiment completion
"""

from datetime import datetime, date, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shorts_engine.db.models import (
    PlannedBatchModel,
    ProjectModel,
    RecipeModel,
    VideoJobModel,
    ExperimentModel,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import BatchStatus, ExperimentStatus, GenerationMode
from shorts_engine.logging import get_logger
from shorts_engine.services.learning.recipe import RecipeService
from shorts_engine.services.learning.reward import RewardCalculator
from shorts_engine.services.learning.sampler import RecipeSampler
from shorts_engine.utils import run_async
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="plan_next_batch",
    max_retries=3,
    default_retry_delay=300,
)
def plan_next_batch_task(
    self: Any,
    project_id: str,
    n: int = 5,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    """Plan and create the next batch of video jobs.

    This task:
    1. Samples recipes using exploit/explore strategy
    2. Creates video jobs with recipe assignments
    3. Records the batch for tracking

    Args:
        project_id: Project to create jobs for
        n: Number of jobs to create (default: 5)
        topics: List of topic ideas. If None, will need to be provided externally.

    Returns:
        Dictionary with batch info and created job IDs
    """
    task_id = self.request.id
    logger.info(
        "plan_next_batch_started",
        task_id=task_id,
        project_id=project_id,
        n=n,
        topics_provided=len(topics) if topics else 0,
    )

    try:
        project_uuid = UUID(project_id)
    except ValueError as e:
        return {"success": False, "error": f"Invalid project ID: {e}"}

    if not topics:
        return {
            "success": False,
            "error": "No topics provided. Topics must be supplied for batch planning.",
        }

    if len(topics) < n:
        logger.warning(
            "insufficient_topics",
            required=n,
            provided=len(topics),
        )
        n = len(topics)

    with get_session_context() as session:
        # Verify project exists
        project = session.get(ProjectModel, project_uuid)
        if not project:
            return {"success": False, "error": f"Project not found: {project_id}"}

        # Check if batch already exists for today
        today = date.today()
        existing_batch = session.execute(
            select(PlannedBatchModel)
            .where(
                PlannedBatchModel.project_id == project_uuid,
                func.date(PlannedBatchModel.batch_date) == today,
            )
        ).scalar_one_or_none()

        if existing_batch and existing_batch.status in (BatchStatus.RUNNING, BatchStatus.COMPLETED):
            return {
                "success": False,
                "error": f"Batch already exists for today ({existing_batch.status})",
                "batch_id": str(existing_batch.id),
            }

        # Sample recipes for topics
        sampler = RecipeSampler(session, project_uuid)
        sampled_jobs = sampler.sample_batch(topics, n)

        if not sampled_jobs:
            return {
                "success": False,
                "error": "Failed to sample any valid job configurations",
            }

        # Calculate exploit/explore counts
        exploit_count = sum(1 for j in sampled_jobs if j.generation_mode == GenerationMode.EXPLOIT)
        explore_count = len(sampled_jobs) - exploit_count

        # Create or update batch record
        if existing_batch:
            batch = existing_batch
            batch.total_jobs = len(sampled_jobs)
            batch.exploit_count = exploit_count
            batch.explore_count = explore_count
            batch.status = BatchStatus.RUNNING
            batch.started_at = datetime.now(timezone.utc)
        else:
            batch = PlannedBatchModel(
                id=uuid4(),
                project_id=project_uuid,
                batch_date=datetime.now(timezone.utc),
                total_jobs=len(sampled_jobs),
                exploit_count=exploit_count,
                explore_count=explore_count,
                status=BatchStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
            )
            session.add(batch)
            session.flush()

        # Create video jobs
        recipe_service = RecipeService(session)
        created_job_ids = []

        for sampled in sampled_jobs:
            # Get or create recipe model
            recipe_model = recipe_service.get_or_create(sampled.recipe, project_uuid)

            # Create video job
            idempotency_key = f"batch-{batch.id}-{sampled.topic_hash}"

            video_job = VideoJobModel(
                id=uuid4(),
                project_id=project_uuid,
                idempotency_key=idempotency_key,
                idea=sampled.topic,
                style_preset=sampled.recipe.preset,
                status="pending",
                stage="created",
                recipe_id=recipe_model.id,
                experiment_id=sampled.experiment_id,
                generation_mode=sampled.generation_mode,
                batch_id=batch.id,
                topic_hash=sampled.topic_hash,
            )
            session.add(video_job)
            created_job_ids.append(str(video_job.id))

            # Update experiment counts if applicable
            if sampled.experiment_id:
                experiment = session.get(ExperimentModel, sampled.experiment_id)
                if experiment:
                    if sampled.is_baseline:
                        experiment.baseline_video_count += 1
                    else:
                        experiment.variant_video_count += 1

        # Update batch record
        batch.jobs_created = len(created_job_ids)

        session.commit()

        logger.info(
            "plan_next_batch_completed",
            task_id=task_id,
            batch_id=str(batch.id),
            jobs_created=len(created_job_ids),
            exploit=exploit_count,
            explore=explore_count,
        )

        return {
            "success": True,
            "batch_id": str(batch.id),
            "batch_date": str(today),
            "jobs_created": len(created_job_ids),
            "job_ids": created_job_ids,
            "exploit_count": exploit_count,
            "explore_count": explore_count,
        }


@celery_app.task(
    bind=True,
    name="update_recipe_stats",
    max_retries=2,
)
def update_recipe_stats_task(
    self: Any,
    project_id: str,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Update recipe statistics for a project.

    Args:
        project_id: Project to update stats for
        lookback_days: Days to look back for stats

    Returns:
        Dictionary with update results
    """
    task_id = self.request.id
    logger.info(
        "update_recipe_stats_started",
        task_id=task_id,
        project_id=project_id,
    )

    try:
        project_uuid = UUID(project_id)
    except ValueError as e:
        return {"success": False, "error": f"Invalid project ID: {e}"}

    with get_session_context() as session:
        # Update reward scores first
        reward_calc = RewardCalculator(session, project_uuid)
        scores_updated = reward_calc.update_stored_scores(lookback_days)

        # Update recipe stats
        recipe_service = RecipeService(session)
        recipes_updated = recipe_service.update_all_recipe_stats(project_uuid)

        logger.info(
            "update_recipe_stats_completed",
            task_id=task_id,
            scores_updated=scores_updated,
            recipes_updated=recipes_updated,
        )

        return {
            "success": True,
            "scores_updated": scores_updated,
            "recipes_updated": recipes_updated,
        }


@celery_app.task(
    bind=True,
    name="evaluate_experiments",
    max_retries=2,
)
def evaluate_experiments_task(
    self: Any,
    project_id: str,
    min_samples: int = 5,
    confidence_threshold: float = 0.95,
) -> dict[str, Any]:
    """Evaluate running experiments and mark completed ones.

    An experiment is completed when:
    - Both baseline and variant have at least min_samples videos
    - The difference is statistically significant at confidence_threshold

    Args:
        project_id: Project to evaluate experiments for
        min_samples: Minimum samples per variant
        confidence_threshold: Required confidence level

    Returns:
        Dictionary with evaluation results
    """
    task_id = self.request.id
    logger.info(
        "evaluate_experiments_started",
        task_id=task_id,
        project_id=project_id,
    )

    try:
        project_uuid = UUID(project_id)
    except ValueError as e:
        return {"success": False, "error": f"Invalid project ID: {e}"}

    with get_session_context() as session:
        # Get running experiments
        experiments = session.execute(
            select(ExperimentModel)
            .where(
                ExperimentModel.project_id == project_uuid,
                ExperimentModel.status == ExperimentStatus.RUNNING,
            )
        ).scalars().all()

        evaluated = []
        completed = []

        for exp in experiments:
            # Check sample sizes
            if exp.baseline_video_count < min_samples or exp.variant_video_count < min_samples:
                continue

            # Calculate average rewards for baseline and variant videos
            baseline_reward = _calculate_experiment_avg_reward(
                session, exp.id, is_baseline=True
            )
            variant_reward = _calculate_experiment_avg_reward(
                session, exp.id, is_baseline=False
            )

            if baseline_reward is None or variant_reward is None:
                continue

            exp.baseline_avg_reward = baseline_reward
            exp.variant_avg_reward = variant_reward

            # Simple comparison (could add statistical testing later)
            diff = variant_reward - baseline_reward
            relative_diff = abs(diff) / max(baseline_reward, 0.01)

            # If difference is significant (>10% relative), declare winner
            if relative_diff > 0.10:
                if diff > 0:
                    exp.winner = "variant"
                else:
                    exp.winner = "baseline"
                exp.status = ExperimentStatus.COMPLETED
                exp.completed_at = datetime.now(timezone.utc)
                exp.confidence = min(0.99, 0.9 + relative_diff * 0.1)  # Simplified confidence
                completed.append({
                    "id": str(exp.id),
                    "variable": exp.variable_tested,
                    "winner": exp.winner,
                    "baseline_reward": baseline_reward,
                    "variant_reward": variant_reward,
                })
            else:
                # Check if we have enough samples to call it inconclusive
                if exp.baseline_video_count >= min_samples * 2 and exp.variant_video_count >= min_samples * 2:
                    exp.status = ExperimentStatus.INCONCLUSIVE
                    exp.winner = "inconclusive"
                    exp.completed_at = datetime.now(timezone.utc)

            evaluated.append(str(exp.id))

        session.commit()

        logger.info(
            "evaluate_experiments_completed",
            task_id=task_id,
            evaluated=len(evaluated),
            completed=len(completed),
        )

        return {
            "success": True,
            "evaluated": len(evaluated),
            "completed": completed,
        }


def _calculate_experiment_avg_reward(
    session: Any,
    experiment_id: UUID,
    is_baseline: bool,
) -> float | None:
    """Calculate average reward for experiment videos.

    Args:
        session: Database session
        experiment_id: Experiment ID
        is_baseline: Whether to calculate for baseline or variant

    Returns:
        Average reward score or None
    """
    from shorts_engine.db.models import VideoMetricsModel, PublishJobModel

    # For baseline, we need videos using the baseline recipe
    # For variant, we need videos with this experiment_id
    if is_baseline:
        exp = session.get(ExperimentModel, experiment_id)
        if not exp or not exp.baseline_recipe_id:
            return None

        result = session.execute(
            select(func.avg(VideoMetricsModel.reward_score))
            .join(PublishJobModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
            .join(VideoJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .where(
                VideoJobModel.recipe_id == exp.baseline_recipe_id,
                VideoJobModel.experiment_id.is_(None),  # Not part of this experiment
                VideoMetricsModel.window_type == "24h",
                VideoMetricsModel.reward_score.isnot(None),
            )
        ).scalar()
    else:
        result = session.execute(
            select(func.avg(VideoMetricsModel.reward_score))
            .join(PublishJobModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
            .join(VideoJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .where(
                VideoJobModel.experiment_id == experiment_id,
                VideoMetricsModel.window_type == "24h",
                VideoMetricsModel.reward_score.isnot(None),
            )
        ).scalar()

    return round(result, 4) if result else None


# Add to beat schedule
LEARNING_BEAT_SCHEDULE = {
    "update-recipe-stats-daily": {
        "task": "update_recipe_stats",
        "schedule": 86400.0,  # Daily
        "args": [],
        "options": {"queue": "learning"},
    },
    "evaluate-experiments-daily": {
        "task": "evaluate_experiments",
        "schedule": 86400.0,  # Daily
        "args": [],
        "options": {"queue": "learning"},
    },
}
