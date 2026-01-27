"""Recipe sampler for exploit/explore video generation.

Implements a multi-armed bandit-inspired approach:
- 70% of jobs use "exploit" mode: sample from top-performing recipes
- 30% of jobs use "explore" mode: mutate ONE variable from a top recipe (A/B test)

Safeguards:
- Prevent duplicate recipe+topic combinations
- Cap extreme values during exploration
- Track experiments for statistical comparison
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shorts_engine.db.models import (
    ExperimentModel,
    VideoJobModel,
)
from shorts_engine.domain.enums import (
    CaptionDensityBucket,
    EndingType,
    ExperimentStatus,
    GenerationMode,
    HookType,
    NarrationWPMBucket,
)
from shorts_engine.logging import get_logger
from shorts_engine.services.learning.recipe import Recipe, RecipeService

logger = get_logger(__name__)


@dataclass
class SampledJob:
    """A job spec ready for creation."""

    recipe: Recipe
    generation_mode: str  # exploit or explore
    topic: str
    topic_hash: str
    experiment_id: UUID | None = None
    is_baseline: bool = False  # True if this is baseline variant in experiment

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recipe": self.recipe.to_dict(),
            "generation_mode": self.generation_mode,
            "topic": self.topic,
            "topic_hash": self.topic_hash,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "is_baseline": self.is_baseline,
        }


class RecipeSampler:
    """Samples recipes for new video jobs using exploit/explore strategy."""

    # Ratio of exploit vs explore
    EXPLOIT_RATIO = 0.7
    EXPLORE_RATIO = 0.3

    # Constraints for exploration
    MIN_SCENE_COUNT = 5
    MAX_SCENE_COUNT = 10

    # Variables that can be mutated during exploration
    MUTABLE_VARIABLES = [
        "preset",
        "hook_type",
        "scene_count",
        "narration_wpm_bucket",
        "caption_density_bucket",
        "ending_type",
    ]

    def __init__(self, session: Session, project_id: UUID):
        """Initialize the sampler.

        Args:
            session: Database session
            project_id: Project to sample for
        """
        self.session = session
        self.project_id = project_id
        self.recipe_service = RecipeService(session)
        self._top_recipes_cache: list[Recipe] | None = None

    def _get_top_recipes(self, k: int = 5) -> list[Recipe]:
        """Get top K performing recipes."""
        if self._top_recipes_cache is None:
            self._top_recipes_cache = self.recipe_service.get_top_recipes(
                self.project_id,
                lookback_days=14,
                min_uses=2,
                limit=k,
            )
        return self._top_recipes_cache

    def _compute_topic_hash(self, topic: str, recipe: Recipe) -> str:
        """Compute a hash for topic+recipe combination.

        Args:
            topic: The video topic/idea
            recipe: The recipe to use

        Returns:
            A hash string for deduplication
        """
        # Normalize topic (lowercase, strip whitespace)
        normalized_topic = topic.lower().strip()
        content = f"{normalized_topic}|{recipe.recipe_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_duplicate(self, topic_hash: str) -> bool:
        """Check if a topic+recipe combination already exists.

        Args:
            topic_hash: The computed hash

        Returns:
            True if this combination was already used
        """
        existing = self.session.execute(
            select(func.count(VideoJobModel.id)).where(
                VideoJobModel.project_id == self.project_id,
                VideoJobModel.topic_hash == topic_hash,
            )
        ).scalar()
        return (existing or 0) > 0

    def _get_mutation_value(self, variable: str, current_value: Any) -> Any:
        """Get a valid mutation value for a variable.

        Args:
            variable: The variable to mutate
            current_value: Current value of the variable

        Returns:
            A new value different from current
        """
        all_values = self.recipe_service.get_recipe_variables()
        possible = all_values.get(variable, [])

        # Filter out current value
        candidates = [v for v in possible if str(v) != str(current_value)]

        if not candidates:
            return current_value

        # For scene_count, apply constraints
        if variable == "scene_count":
            int_candidates = [
                int(v) for v in candidates if self.MIN_SCENE_COUNT <= int(v) <= self.MAX_SCENE_COUNT
            ]
            if not int_candidates:
                return current_value
            return random.choice(int_candidates)

        return random.choice(candidates)

    def _sample_exploit(self, topic: str) -> SampledJob | None:
        """Sample an exploit job (use top-performing recipe).

        Args:
            topic: The video topic

        Returns:
            SampledJob or None if no valid recipe found
        """
        top_recipes = self._get_top_recipes()

        if not top_recipes:
            logger.warning("no_top_recipes_for_exploit", project_id=str(self.project_id))
            return None

        # Weighted sampling based on reward score
        weights = []
        for r in top_recipes:
            # Use reward score as weight, with minimum of 0.1
            weight = max(r.avg_reward_score or 0.5, 0.1)
            weights.append(weight)

        recipe = random.choices(top_recipes, weights=weights, k=1)[0]
        topic_hash = self._compute_topic_hash(topic, recipe)

        # Check for duplicates
        if self._is_duplicate(topic_hash):
            # Try other recipes
            for alt_recipe in top_recipes:
                if alt_recipe.recipe_hash != recipe.recipe_hash:
                    alt_topic_hash = self._compute_topic_hash(topic, alt_recipe)
                    if not self._is_duplicate(alt_topic_hash):
                        recipe = alt_recipe
                        topic_hash = alt_topic_hash
                        break
            else:
                logger.warning(
                    "all_exploit_recipes_duplicated",
                    topic=topic[:50],
                    project_id=str(self.project_id),
                )
                return None

        return SampledJob(
            recipe=recipe,
            generation_mode=GenerationMode.EXPLOIT,
            topic=topic,
            topic_hash=topic_hash,
        )

    def _sample_explore(self, topic: str) -> SampledJob | None:
        """Sample an explore job (mutate one variable from top recipe).

        Args:
            topic: The video topic

        Returns:
            SampledJob or None if no valid mutation found
        """
        top_recipes = self._get_top_recipes()

        if not top_recipes:
            # If no top recipes, create a default explore recipe
            logger.info("creating_default_explore_recipe", project_id=str(self.project_id))
            base_recipe = Recipe(
                preset="DARK_DYSTOPIAN_ANIME",
                hook_type=HookType.STATEMENT,
                scene_count=7,
                narration_wpm_bucket=NarrationWPMBucket.MEDIUM,
                caption_density_bucket=CaptionDensityBucket.MEDIUM,
                ending_type=EndingType.RESOLVE,
            )
        else:
            # Pick a random top recipe as base
            base_recipe = random.choice(top_recipes)

        # Pick a random variable to mutate
        variable = random.choice(self.MUTABLE_VARIABLES)
        current_value = getattr(base_recipe, variable)
        new_value = self._get_mutation_value(variable, current_value)

        # Create mutated recipe
        mutated_recipe = base_recipe.mutate(variable, new_value)
        topic_hash = self._compute_topic_hash(topic, mutated_recipe)

        # Check for duplicates
        if self._is_duplicate(topic_hash):
            # Try different mutations
            for alt_variable in self.MUTABLE_VARIABLES:
                if alt_variable != variable:
                    alt_value = self._get_mutation_value(
                        alt_variable, getattr(base_recipe, alt_variable)
                    )
                    alt_recipe = base_recipe.mutate(alt_variable, alt_value)
                    alt_topic_hash = self._compute_topic_hash(topic, alt_recipe)
                    if not self._is_duplicate(alt_topic_hash):
                        mutated_recipe = alt_recipe
                        variable = alt_variable
                        new_value = alt_value
                        topic_hash = alt_topic_hash
                        break
            else:
                logger.warning(
                    "all_explore_mutations_duplicated",
                    topic=topic[:50],
                    project_id=str(self.project_id),
                )
                return None

        # Create or find an experiment for this mutation
        experiment = self._get_or_create_experiment(
            base_recipe=base_recipe,
            variable=variable,
            baseline_value=str(current_value),
            variant_value=str(new_value),
        )

        return SampledJob(
            recipe=mutated_recipe,
            generation_mode=GenerationMode.EXPLORE,
            topic=topic,
            topic_hash=topic_hash,
            experiment_id=experiment.id if experiment else None,
            is_baseline=False,
        )

    def _get_or_create_experiment(
        self,
        base_recipe: Recipe,
        variable: str,
        baseline_value: str,
        variant_value: str,
    ) -> ExperimentModel | None:
        """Get or create an experiment for this mutation.

        Args:
            base_recipe: The base recipe
            variable: Variable being tested
            baseline_value: Original value
            variant_value: Mutated value

        Returns:
            Experiment model or None
        """
        # Look for existing running experiment with same parameters
        existing = self.session.execute(
            select(ExperimentModel).where(
                ExperimentModel.project_id == self.project_id,
                ExperimentModel.variable_tested == variable,
                ExperimentModel.baseline_value == baseline_value,
                ExperimentModel.variant_value == variant_value,
                ExperimentModel.status == ExperimentStatus.RUNNING,
            )
        ).scalar_one_or_none()

        if existing:
            return existing

        # Get or create the baseline recipe model
        base_recipe_model = self.recipe_service.get_or_create(base_recipe, self.project_id)

        # Create new experiment
        experiment = ExperimentModel(
            id=uuid4(),
            project_id=self.project_id,
            name=f"{variable}: {baseline_value} vs {variant_value}",
            description=f"Testing whether changing {variable} from {baseline_value} to {variant_value} improves performance",
            variable_tested=variable,
            baseline_recipe_id=base_recipe_model.id,
            baseline_value=baseline_value,
            variant_value=variant_value,
            status=ExperimentStatus.RUNNING,
        )
        self.session.add(experiment)
        self.session.flush()

        logger.info(
            "experiment_created",
            experiment_id=str(experiment.id),
            variable=variable,
            baseline=baseline_value,
            variant=variant_value,
        )

        return experiment

    def sample(self, topic: str) -> SampledJob | None:
        """Sample a job for the given topic.

        Args:
            topic: The video topic/idea

        Returns:
            SampledJob or None if sampling failed
        """
        # Decide exploit vs explore
        if random.random() < self.EXPLOIT_RATIO:
            return self._sample_exploit(topic)
        else:
            return self._sample_explore(topic)

    def sample_batch(
        self,
        topics: list[str],
        n: int | None = None,
    ) -> list[SampledJob]:
        """Sample a batch of jobs with controlled exploit/explore ratio.

        Args:
            topics: List of available topics
            n: Number of jobs to create (defaults to len(topics))

        Returns:
            List of SampledJob specs
        """
        if not topics:
            return []

        n = n or len(topics)
        n = min(n, len(topics))

        # Calculate exact split
        exploit_count = int(n * self.EXPLOIT_RATIO)
        explore_count = n - exploit_count

        # Shuffle topics
        available_topics = topics.copy()
        random.shuffle(available_topics)

        jobs: list[SampledJob] = []
        used_hashes: set[str] = set()

        # Generate exploit jobs
        for _ in range(exploit_count):
            if not available_topics:
                break

            topic = available_topics.pop()
            job = self._sample_exploit(topic)

            if job and job.topic_hash not in used_hashes:
                jobs.append(job)
                used_hashes.add(job.topic_hash)
            else:
                # Put topic back if we couldn't use it
                available_topics.insert(0, topic)

        # Generate explore jobs
        for _ in range(explore_count):
            if not available_topics:
                break

            topic = available_topics.pop()
            job = self._sample_explore(topic)

            if job and job.topic_hash not in used_hashes:
                jobs.append(job)
                used_hashes.add(job.topic_hash)
            else:
                # Put topic back if we couldn't use it
                available_topics.insert(0, topic)

        logger.info(
            "batch_sampled",
            project_id=str(self.project_id),
            total=len(jobs),
            exploit=[j for j in jobs if j.generation_mode == GenerationMode.EXPLOIT],
            explore=[j for j in jobs if j.generation_mode == GenerationMode.EXPLORE],
        )

        return jobs

    def get_recommendations(
        self,
        n: int = 5,
        topics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get recipe recommendations for new videos.

        Args:
            n: Number of recommendations
            topics: Optional topics to generate recommendations for

        Returns:
            List of recommendation dictionaries
        """
        # If no topics provided, return recipe recommendations only
        if not topics:
            top_recipes = self._get_top_recipes(k=n)
            return [
                {
                    "type": "exploit",
                    "recipe": r.to_dict(),
                    "reason": f"Top performer with {r.avg_reward_score:.2f} avg reward ({r.times_used} videos)",
                }
                for r in top_recipes
            ]

        # Sample jobs for provided topics
        jobs = self.sample_batch(topics[:n], n)

        recommendations = []
        for job in jobs:
            rec = {
                "type": job.generation_mode,
                "recipe": job.recipe.to_dict(),
                "topic": job.topic,
            }

            if job.generation_mode == GenerationMode.EXPLOIT:
                rec["reason"] = "Based on top-performing recipe"
            else:
                rec["reason"] = "A/B test mutation"
                if job.experiment_id:
                    rec["experiment_id"] = str(job.experiment_id)

            recommendations.append(rec)

        return recommendations
