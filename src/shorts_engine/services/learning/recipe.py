"""Recipe service for managing video production recipes.

A recipe defines the key parameters that influence video performance:
- preset: Visual style preset
- hook_type: How the video opens
- scene_count: Number of scenes
- narration_wpm_bucket: Narration speed (slow/medium/fast)
- caption_density_bucket: Caption frequency (sparse/medium/dense)
- ending_type: How the video ends (cliffhanger/resolve)
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from shorts_engine.db.models import (
    PublishJobModel,
    RecipeModel,
    VideoJobModel,
    VideoMetricsModel,
    VideoRecipeFeaturesModel,
)
from shorts_engine.domain.enums import (
    CaptionDensityBucket,
    EndingType,
    HookType,
    NarrationWPMBucket,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Recipe:
    """A video production recipe."""

    preset: str
    hook_type: str
    scene_count: int
    narration_wpm_bucket: str
    caption_density_bucket: str
    ending_type: str

    # Optional metadata
    id: UUID | None = None
    project_id: UUID | None = None
    times_used: int = 0
    avg_reward_score: float | None = None
    best_reward_score: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize values."""
        # Ensure enums are valid strings
        self.hook_type = str(self.hook_type)
        self.ending_type = str(self.ending_type)
        self.narration_wpm_bucket = str(self.narration_wpm_bucket)
        self.caption_density_bucket = str(self.caption_density_bucket)

    @property
    def recipe_hash(self) -> str:
        """Generate a deterministic hash for this recipe."""
        components = [
            self.preset,
            self.hook_type,
            str(self.scene_count),
            self.narration_wpm_bucket,
            self.caption_density_bucket,
            self.ending_type,
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id) if self.id else None,
            "preset": self.preset,
            "hook_type": self.hook_type,
            "scene_count": self.scene_count,
            "narration_wpm_bucket": self.narration_wpm_bucket,
            "caption_density_bucket": self.caption_density_bucket,
            "ending_type": self.ending_type,
            "recipe_hash": self.recipe_hash,
            "times_used": self.times_used,
            "avg_reward_score": self.avg_reward_score,
            "best_reward_score": self.best_reward_score,
        }

    def mutate(self, variable: str, new_value: Any) -> "Recipe":
        """Create a new recipe with one variable changed.

        Args:
            variable: Which variable to change
            new_value: The new value for that variable

        Returns:
            A new Recipe with the mutation applied
        """
        values = {
            "preset": self.preset,
            "hook_type": self.hook_type,
            "scene_count": self.scene_count,
            "narration_wpm_bucket": self.narration_wpm_bucket,
            "caption_density_bucket": self.caption_density_bucket,
            "ending_type": self.ending_type,
        }

        if variable not in values:
            raise ValueError(f"Unknown recipe variable: {variable}")

        values[variable] = new_value
        return Recipe(**values)  # type: ignore[arg-type]

    @classmethod
    def from_model(cls, model: RecipeModel) -> "Recipe":
        """Create from ORM model."""
        return cls(
            id=model.id,
            project_id=model.project_id,
            preset=model.preset,
            hook_type=model.hook_type,
            scene_count=model.scene_count,
            narration_wpm_bucket=model.narration_wpm_bucket,
            caption_density_bucket=model.caption_density_bucket,
            ending_type=model.ending_type,
            times_used=model.times_used or 0,
            avg_reward_score=model.avg_reward_score,
            best_reward_score=model.best_reward_score,
        )

    @classmethod
    def from_video_features(cls, features: VideoRecipeFeaturesModel) -> "Recipe":
        """Extract recipe from video features."""
        return cls(
            preset=features.style_preset or "DARK_DYSTOPIAN_ANIME",
            hook_type=features.hook_type or HookType.STATEMENT,
            scene_count=features.scene_count or 7,
            narration_wpm_bucket=NarrationWPMBucket.from_wpm(features.narration_wpm),
            caption_density_bucket=CaptionDensityBucket.from_density(features.caption_density),
            ending_type=features.ending_type or EndingType.RESOLVE,
        )


class RecipeService:
    """Service for managing recipes and their performance data."""

    def __init__(self, session: Session):
        """Initialize the service.

        Args:
            session: Database session
        """
        self.session = session

    def get_or_create(self, recipe: Recipe, project_id: UUID) -> RecipeModel:
        """Get an existing recipe or create a new one.

        Args:
            recipe: The recipe to find or create
            project_id: The project this recipe belongs to

        Returns:
            The ORM model for this recipe
        """
        recipe_hash = recipe.recipe_hash

        existing = self.session.execute(
            select(RecipeModel).where(RecipeModel.recipe_hash == recipe_hash)
        ).scalar_one_or_none()

        if existing:
            return existing

        # Create new recipe
        model = RecipeModel(
            id=uuid4(),
            project_id=project_id,
            preset=recipe.preset,
            hook_type=recipe.hook_type,
            scene_count=recipe.scene_count,
            narration_wpm_bucket=recipe.narration_wpm_bucket,
            caption_density_bucket=recipe.caption_density_bucket,
            ending_type=recipe.ending_type,
            recipe_hash=recipe_hash,
        )
        self.session.add(model)
        self.session.flush()

        logger.info(
            "recipe_created",
            recipe_id=str(model.id),
            recipe_hash=recipe_hash,
            project_id=str(project_id),
        )

        return model

    def get_top_recipes(
        self,
        project_id: UUID,
        lookback_days: int = 14,
        min_uses: int = 2,
        limit: int = 10,
    ) -> list[Recipe]:
        """Get top-performing recipes for a project.

        Args:
            project_id: Project to get recipes for
            lookback_days: Only consider recipes used within this window
            min_uses: Minimum number of uses to be considered
            limit: Maximum number of recipes to return

        Returns:
            List of top recipes sorted by avg_reward_score
        """
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        recipes = (
            self.session.execute(
                select(RecipeModel)
                .where(
                    RecipeModel.project_id == project_id,
                    RecipeModel.times_used >= min_uses,
                    RecipeModel.last_used_at >= cutoff,
                    RecipeModel.avg_reward_score.isnot(None),
                )
                .order_by(desc(RecipeModel.avg_reward_score))
                .limit(limit)
            )
            .scalars()
            .all()
        )

        return [Recipe.from_model(r) for r in recipes]

    def update_recipe_stats(self, recipe_id: UUID) -> None:
        """Update aggregate statistics for a recipe.

        Args:
            recipe_id: Recipe to update
        """
        recipe = self.session.get(RecipeModel, recipe_id)
        if not recipe:
            return

        # Get all video jobs using this recipe that have metrics
        stats = self.session.execute(
            select(
                func.count(VideoJobModel.id).label("count"),
                func.avg(VideoMetricsModel.reward_score).label("avg_reward"),
                func.max(VideoMetricsModel.reward_score).label("best_reward"),
                func.max(VideoJobModel.created_at).label("last_used"),
            )
            .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
            .where(
                VideoJobModel.recipe_id == recipe_id,
                VideoMetricsModel.window_type == "24h",  # Use 24h window for recipe stats
                VideoMetricsModel.reward_score.isnot(None),
            )
        ).first()

        if stats and stats.count:  # type: ignore[truthy-function]
            recipe.times_used = stats.count  # type: ignore[assignment]
            recipe.avg_reward_score = round(stats.avg_reward, 4) if stats.avg_reward else None
            recipe.best_reward_score = round(stats.best_reward, 4) if stats.best_reward else None
            recipe.last_used_at = stats.last_used

        self.session.commit()

        logger.info(
            "recipe_stats_updated",
            recipe_id=str(recipe_id),
            times_used=recipe.times_used,
            avg_reward=recipe.avg_reward_score,
        )

    def update_all_recipe_stats(self, project_id: UUID) -> int:
        """Update stats for all recipes in a project.

        Args:
            project_id: Project to update recipes for

        Returns:
            Number of recipes updated
        """
        recipes = (
            self.session.execute(select(RecipeModel).where(RecipeModel.project_id == project_id))
            .scalars()
            .all()
        )

        count = 0
        for recipe in recipes:
            self.update_recipe_stats(recipe.id)
            count += 1

        logger.info(
            "all_recipe_stats_updated",
            project_id=str(project_id),
            recipes_updated=count,
        )

        return count

    def extract_recipe_from_job(self, video_job_id: UUID) -> Recipe | None:
        """Extract recipe from an existing video job's features.

        Args:
            video_job_id: Video job to extract recipe from

        Returns:
            Recipe if features exist, None otherwise
        """
        features = self.session.execute(
            select(VideoRecipeFeaturesModel).where(
                VideoRecipeFeaturesModel.video_job_id == video_job_id
            )
        ).scalar_one_or_none()

        if not features:
            return None

        return Recipe.from_video_features(features)

    def backfill_recipes_from_features(self, project_id: UUID) -> int:
        """Backfill recipe assignments for existing videos.

        Scans videos that don't have a recipe_id but do have features,
        extracts their recipe, and assigns it.

        Args:
            project_id: Project to backfill

        Returns:
            Number of videos updated
        """
        # Find videos with features but no recipe
        jobs = (
            self.session.execute(
                select(VideoJobModel)
                .join(
                    VideoRecipeFeaturesModel,
                    VideoRecipeFeaturesModel.video_job_id == VideoJobModel.id,
                )
                .where(
                    VideoJobModel.project_id == project_id,
                    VideoJobModel.recipe_id.is_(None),
                )
            )
            .scalars()
            .all()
        )

        count = 0
        for job in jobs:
            recipe = self.extract_recipe_from_job(job.id)
            if recipe:
                model = self.get_or_create(recipe, project_id)
                job.recipe_id = model.id
                job.generation_mode = "manual"  # Existing videos were manually created
                count += 1

        self.session.commit()

        logger.info(
            "recipes_backfilled",
            project_id=str(project_id),
            videos_updated=count,
        )

        return count

    def get_recipe_variables(self) -> dict[str, list[str]]:
        """Get all possible values for each recipe variable.

        Returns:
            Dictionary mapping variable name to list of possible values
        """
        from shorts_engine.presets.styles import get_preset_names

        return {
            "preset": get_preset_names(),
            "hook_type": [h.value for h in HookType],
            "scene_count": ["5", "6", "7", "8", "9", "10"],
            "narration_wpm_bucket": [n.value for n in NarrationWPMBucket],
            "caption_density_bucket": [c.value for c in CaptionDensityBucket],
            "ending_type": [e.value for e in EndingType],
        }
