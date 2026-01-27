"""Admin endpoints for learning loop management."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from shorts_engine.db.models import (
    ExperimentModel,
    PlannedBatchModel,
    ProjectModel,
    PublishJobModel,
    RecipeModel,
    VideoJobModel,
    VideoMetricsModel,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import ExperimentStatus
from shorts_engine.logging import get_logger
from shorts_engine.services.learning.recipe import RecipeService
from shorts_engine.services.learning.reward import RewardCalculator
from shorts_engine.services.learning.sampler import RecipeSampler

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = get_logger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class RecipeResponse(BaseModel):
    """Recipe details."""

    id: str
    preset: str
    hook_type: str
    scene_count: int
    narration_wpm_bucket: str
    caption_density_bucket: str
    ending_type: str
    recipe_hash: str
    times_used: int
    avg_reward_score: float | None
    best_reward_score: float | None
    last_used_at: datetime | None


class ExperimentResponse(BaseModel):
    """Experiment details."""

    id: str
    name: str
    variable_tested: str
    baseline_value: str
    variant_value: str
    status: str
    baseline_video_count: int
    variant_video_count: int
    baseline_avg_reward: float | None
    variant_avg_reward: float | None
    winner: str | None
    confidence: float | None
    started_at: datetime
    completed_at: datetime | None


class BatchResponse(BaseModel):
    """Batch details."""

    id: str
    batch_date: datetime
    total_jobs: int
    exploit_count: int
    explore_count: int
    status: str
    jobs_created: int
    jobs_completed: int
    created_at: datetime


class RecommendationResponse(BaseModel):
    """A recipe recommendation."""

    type: str  # exploit or explore
    recipe: dict[str, Any]
    topic: str | None = None
    reason: str
    experiment_id: str | None = None


class RecommendationsResponse(BaseModel):
    """List of recommendations."""

    recommendations: list[RecommendationResponse]
    top_recipes: list[RecipeResponse]
    running_experiments: int
    recent_batches: int


class StatsResponse(BaseModel):
    """Learning loop statistics."""

    total_recipes: int
    total_experiments: int
    running_experiments: int
    completed_experiments: int
    total_batches: int
    videos_last_7d: int
    avg_reward_score: float | None
    top_recipe: RecipeResponse | None


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    summary="Get recipe recommendations",
    description="Get recommendations for next video batch based on learning loop data.",
)
async def get_recommendations(
    project_id: str = Query(..., description="Project ID"),
    n: int = Query(5, ge=1, le=20, description="Number of recommendations"),
    topics: str | None = Query(None, description="Comma-separated list of topics"),
) -> RecommendationsResponse:
    """Get recommendations for the next batch of videos."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    topic_list = [t.strip() for t in topics.split(",")] if topics else None

    with get_session_context() as session:
        # Verify project exists
        project = session.get(ProjectModel, project_uuid)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )

        # Get recommendations
        sampler = RecipeSampler(session, project_uuid)
        recs = sampler.get_recommendations(n=n, topics=topic_list)

        # Get top recipes
        recipe_service = RecipeService(session)
        top_recipes = recipe_service.get_top_recipes(project_uuid, limit=5)

        # Count running experiments
        running_experiments = (
            session.execute(
                select(func.count())
                .select_from(ExperimentModel)
                .where(
                    ExperimentModel.project_id == project_uuid,
                    ExperimentModel.status == ExperimentStatus.RUNNING,
                )
            ).scalar()
            or 0
        )

        # Count recent batches
        cutoff = datetime.now(UTC) - timedelta(days=7)
        recent_batches = (
            session.execute(
                select(func.count())
                .select_from(PlannedBatchModel)
                .where(
                    PlannedBatchModel.project_id == project_uuid,
                    PlannedBatchModel.created_at >= cutoff,
                )
            ).scalar()
            or 0
        )

        return RecommendationsResponse(
            recommendations=[
                RecommendationResponse(
                    type=r.get("type", "exploit"),
                    recipe=r.get("recipe", {}),
                    topic=r.get("topic"),
                    reason=r.get("reason", ""),
                    experiment_id=r.get("experiment_id"),
                )
                for r in recs
            ],
            top_recipes=[
                RecipeResponse(
                    id=str(r.id) if r.id else "",
                    preset=r.preset,
                    hook_type=r.hook_type,
                    scene_count=r.scene_count,
                    narration_wpm_bucket=r.narration_wpm_bucket,
                    caption_density_bucket=r.caption_density_bucket,
                    ending_type=r.ending_type,
                    recipe_hash=r.recipe_hash,
                    times_used=r.times_used,
                    avg_reward_score=r.avg_reward_score,
                    best_reward_score=r.best_reward_score,
                    last_used_at=None,
                )
                for r in top_recipes
            ],
            running_experiments=running_experiments,
            recent_batches=recent_batches,
        )


@router.get(
    "/recipes",
    response_model=list[RecipeResponse],
    summary="List recipes",
    description="List all recipes for a project.",
)
async def list_recipes(
    project_id: str = Query(..., description="Project ID"),
    sort_by: str = Query(
        "avg_reward_score", description="Sort by: avg_reward_score, times_used, created_at"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
) -> list[RecipeResponse]:
    """List recipes for a project."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    with get_session_context() as session:
        query = select(RecipeModel).where(RecipeModel.project_id == project_uuid)

        if sort_by == "avg_reward_score":
            query = query.order_by(desc(RecipeModel.avg_reward_score).nullslast())
        elif sort_by == "times_used":
            query = query.order_by(desc(RecipeModel.times_used))
        else:
            query = query.order_by(desc(RecipeModel.created_at))

        query = query.limit(limit)

        recipes = session.execute(query).scalars().all()

        return [
            RecipeResponse(
                id=str(r.id),
                preset=r.preset,
                hook_type=r.hook_type,
                scene_count=r.scene_count,
                narration_wpm_bucket=r.narration_wpm_bucket,
                caption_density_bucket=r.caption_density_bucket,
                ending_type=r.ending_type,
                recipe_hash=r.recipe_hash,
                times_used=r.times_used or 0,
                avg_reward_score=r.avg_reward_score,
                best_reward_score=r.best_reward_score,
                last_used_at=r.last_used_at,
            )
            for r in recipes
        ]


@router.get(
    "/experiments",
    response_model=list[ExperimentResponse],
    summary="List experiments",
    description="List A/B test experiments for a project.",
)
async def list_experiments(
    project_id: str = Query(..., description="Project ID"),
    status_filter: str | None = Query(
        None, description="Filter by status: running, completed, inconclusive"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
) -> list[ExperimentResponse]:
    """List experiments for a project."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    with get_session_context() as session:
        query = select(ExperimentModel).where(ExperimentModel.project_id == project_uuid)

        if status_filter:
            query = query.where(ExperimentModel.status == status_filter)

        query = query.order_by(desc(ExperimentModel.created_at)).limit(limit)

        experiments = session.execute(query).scalars().all()

        return [
            ExperimentResponse(
                id=str(e.id),
                name=e.name,
                variable_tested=e.variable_tested,
                baseline_value=e.baseline_value,
                variant_value=e.variant_value,
                status=e.status,
                baseline_video_count=e.baseline_video_count,
                variant_video_count=e.variant_video_count,
                baseline_avg_reward=e.baseline_avg_reward,
                variant_avg_reward=e.variant_avg_reward,
                winner=e.winner,
                confidence=e.confidence,
                started_at=e.started_at,
                completed_at=e.completed_at,
            )
            for e in experiments
        ]


@router.get(
    "/batches",
    response_model=list[BatchResponse],
    summary="List planned batches",
    description="List planned video batches for a project.",
)
async def list_batches(
    project_id: str = Query(..., description="Project ID"),
    limit: int = Query(10, ge=1, le=50, description="Number of results"),
) -> list[BatchResponse]:
    """List batches for a project."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    with get_session_context() as session:
        batches = (
            session.execute(
                select(PlannedBatchModel)
                .where(PlannedBatchModel.project_id == project_uuid)
                .order_by(desc(PlannedBatchModel.created_at))
                .limit(limit)
            )
            .scalars()
            .all()
        )

        return [
            BatchResponse(
                id=str(b.id),
                batch_date=b.batch_date,
                total_jobs=b.total_jobs,
                exploit_count=b.exploit_count,
                explore_count=b.explore_count,
                status=b.status,
                jobs_created=b.jobs_created,
                jobs_completed=b.jobs_completed,
                created_at=b.created_at,
            )
            for b in batches
        ]


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get learning loop stats",
    description="Get aggregate statistics for the learning loop.",
)
async def get_learning_stats(
    project_id: str = Query(..., description="Project ID"),
) -> StatsResponse:
    """Get learning loop statistics."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    with get_session_context() as session:
        # Count recipes
        total_recipes = (
            session.execute(
                select(func.count())
                .select_from(RecipeModel)
                .where(RecipeModel.project_id == project_uuid)
            ).scalar()
            or 0
        )

        # Count experiments
        total_experiments = (
            session.execute(
                select(func.count())
                .select_from(ExperimentModel)
                .where(ExperimentModel.project_id == project_uuid)
            ).scalar()
            or 0
        )

        running_experiments = (
            session.execute(
                select(func.count())
                .select_from(ExperimentModel)
                .where(
                    ExperimentModel.project_id == project_uuid,
                    ExperimentModel.status == ExperimentStatus.RUNNING,
                )
            ).scalar()
            or 0
        )

        completed_experiments = (
            session.execute(
                select(func.count())
                .select_from(ExperimentModel)
                .where(
                    ExperimentModel.project_id == project_uuid,
                    ExperimentModel.status == ExperimentStatus.COMPLETED,
                )
            ).scalar()
            or 0
        )

        # Count batches
        total_batches = (
            session.execute(
                select(func.count())
                .select_from(PlannedBatchModel)
                .where(PlannedBatchModel.project_id == project_uuid)
            ).scalar()
            or 0
        )

        # Count videos in last 7 days
        cutoff = datetime.now(UTC) - timedelta(days=7)
        videos_last_7d = (
            session.execute(
                select(func.count())
                .select_from(VideoJobModel)
                .where(
                    VideoJobModel.project_id == project_uuid,
                    VideoJobModel.created_at >= cutoff,
                )
            ).scalar()
            or 0
        )

        # Average reward score
        avg_reward = session.execute(
            select(func.avg(VideoMetricsModel.reward_score))
            .join(PublishJobModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
            .join(VideoJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .where(
                VideoJobModel.project_id == project_uuid,
                VideoMetricsModel.window_type == "24h",
                VideoMetricsModel.reward_score.isnot(None),
            )
        ).scalar()

        # Top recipe
        top_recipe_model = session.execute(
            select(RecipeModel)
            .where(
                RecipeModel.project_id == project_uuid,
                RecipeModel.avg_reward_score.isnot(None),
            )
            .order_by(desc(RecipeModel.avg_reward_score))
            .limit(1)
        ).scalar_one_or_none()

        top_recipe = None
        if top_recipe_model:
            top_recipe = RecipeResponse(
                id=str(top_recipe_model.id),
                preset=top_recipe_model.preset,
                hook_type=top_recipe_model.hook_type,
                scene_count=top_recipe_model.scene_count,
                narration_wpm_bucket=top_recipe_model.narration_wpm_bucket,
                caption_density_bucket=top_recipe_model.caption_density_bucket,
                ending_type=top_recipe_model.ending_type,
                recipe_hash=top_recipe_model.recipe_hash,
                times_used=top_recipe_model.times_used or 0,
                avg_reward_score=top_recipe_model.avg_reward_score,
                best_reward_score=top_recipe_model.best_reward_score,
                last_used_at=top_recipe_model.last_used_at,
            )

        return StatsResponse(
            total_recipes=total_recipes,
            total_experiments=total_experiments,
            running_experiments=running_experiments,
            completed_experiments=completed_experiments,
            total_batches=total_batches,
            videos_last_7d=videos_last_7d,
            avg_reward_score=round(avg_reward, 4) if avg_reward else None,
            top_recipe=top_recipe,
        )


@router.post(
    "/backfill-recipes",
    summary="Backfill recipes from existing videos",
    description="Extract recipes from existing videos that don't have recipe assignments.",
)
async def backfill_recipes(
    project_id: str = Query(..., description="Project ID"),
) -> dict[str, Any]:
    """Backfill recipes from existing video features."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    with get_session_context() as session:
        recipe_service = RecipeService(session)
        count = recipe_service.backfill_recipes_from_features(project_uuid)

        return {
            "success": True,
            "videos_updated": count,
        }


@router.post(
    "/update-stats",
    summary="Update recipe stats",
    description="Trigger an update of recipe statistics.",
)
async def update_stats(
    project_id: str = Query(..., description="Project ID"),
    lookback_days: int = Query(30, ge=1, le=90, description="Days to look back"),
) -> dict[str, Any]:
    """Update recipe statistics."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID",
        )

    with get_session_context() as session:
        # Update reward scores
        reward_calc = RewardCalculator(session, project_uuid)
        scores_updated = reward_calc.update_stored_scores(lookback_days)

        # Update recipe stats
        recipe_service = RecipeService(session)
        recipes_updated = recipe_service.update_all_recipe_stats(project_uuid)

        return {
            "success": True,
            "scores_updated": scores_updated,
            "recipes_updated": recipes_updated,
        }
