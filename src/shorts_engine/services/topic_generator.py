"""Topic Generator service for autonomous video ideation.

Builds context from historical performance data and uses topic providers
to generate new, high-potential video topics.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from shorts_engine.adapters.llm import LLMProvider
from shorts_engine.adapters.topics import (
    GeneratedTopic,
    LLMTopicProvider,
    StubTopicProvider,
    TopicContext,
    TopicProvider,
)
from shorts_engine.config import get_settings
from shorts_engine.db.models import (
    ProjectModel,
    PublishJobModel,
    RecipeModel,
    VideoJobModel,
    VideoMetricsModel,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class TopicGenerator:
    """Service for generating video topics using historical context.

    Aggregates performance data from the database and uses topic providers
    to generate new, potentially viral topics.
    """

    def __init__(
        self,
        session: Session,
        topic_provider: TopicProvider | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        """Initialize the topic generator.

        Args:
            session: Database session
            topic_provider: Optional topic provider. If not provided, will be
                           created based on config settings.
            llm_provider: Optional LLM provider for LLM-based topic generation.
        """
        self.session = session
        self._topic_provider = topic_provider
        self._llm_provider = llm_provider

    @property
    def topic_provider(self) -> TopicProvider:
        """Get or create the topic provider."""
        if self._topic_provider is None:
            settings = get_settings()
            if settings.topic_provider == "llm":
                if self._llm_provider is None:
                    raise ValueError(
                        "LLM provider required for llm topic provider. "
                        "Pass llm_provider to TopicGenerator or use topic_provider=stub"
                    )
                self._topic_provider = LLMTopicProvider(self._llm_provider)
            else:
                self._topic_provider = StubTopicProvider()
        return self._topic_provider

    def build_context(
        self,
        project_id: UUID,
        lookback_days: int = 30,
        top_n: int = 10,
        recent_n: int = 20,
    ) -> TopicContext:
        """Build context for topic generation from database.

        Args:
            project_id: Project to build context for
            lookback_days: Days to look back for performance data
            top_n: Number of top-performing topics to include
            recent_n: Number of recent topics to include (for deduplication)

        Returns:
            TopicContext with aggregated data
        """
        project = self.session.get(ProjectModel, project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Get top-performing topics by reward score
        top_performing = self._get_top_performing_topics(project_id, cutoff, top_n)

        # Get recent topics to avoid
        recent_topics = self._get_recent_topics(project_id, recent_n)

        # Get top hook types by average reward
        top_hooks = self._get_top_hook_types(project_id, cutoff)

        # Get average video duration
        avg_duration = self._get_avg_video_duration(project_id, cutoff)

        # Extract niche from project settings if available
        niche = None
        if project.settings:
            niche = project.settings.get("niche")

        context = TopicContext(
            project_id=project_id,
            project_name=project.name,
            project_description=project.description,
            niche=niche,
            top_performing_topics=top_performing,
            recent_topics=recent_topics,
            top_hook_types=top_hooks,
            avg_video_duration_seconds=avg_duration,
        )

        logger.info(
            "topic_context_built",
            project_id=str(project_id),
            top_performing_count=len(top_performing),
            recent_topics_count=len(recent_topics),
            top_hooks=top_hooks,
        )

        return context

    def _get_top_performing_topics(
        self,
        project_id: UUID,
        cutoff: datetime,
        limit: int,
    ) -> list[str]:
        """Get top-performing video topics by reward score."""
        results = (
            self.session.execute(
                select(VideoJobModel.idea, func.max(VideoMetricsModel.reward_score).label("score"))
                .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
                .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
                .where(
                    VideoJobModel.project_id == project_id,
                    VideoJobModel.created_at >= cutoff,
                    VideoMetricsModel.window_type == "24h",
                    VideoMetricsModel.reward_score.isnot(None),
                )
                .group_by(VideoJobModel.idea)
                .order_by(desc("score"))
                .limit(limit)
            )
            .mappings()
            .all()
        )

        return [r["idea"] for r in results if r["idea"]]

    def _get_recent_topics(self, project_id: UUID, limit: int) -> list[str]:
        """Get recent video topics to avoid duplicates."""
        results = (
            self.session.execute(
                select(VideoJobModel.idea)
                .where(
                    VideoJobModel.project_id == project_id,
                    VideoJobModel.idea.isnot(None),
                )
                .order_by(desc(VideoJobModel.created_at))
                .limit(limit)
            )
            .scalars()
            .all()
        )

        return list(results)

    def _get_top_hook_types(self, project_id: UUID, cutoff: datetime) -> list[str]:
        """Get hook types ranked by average reward score."""
        results = (
            self.session.execute(
                select(
                    RecipeModel.hook_type,
                    func.avg(VideoMetricsModel.reward_score).label("avg_score"),
                )
                .join(VideoJobModel, VideoJobModel.recipe_id == RecipeModel.id)
                .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
                .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
                .where(
                    RecipeModel.project_id == project_id,
                    VideoJobModel.created_at >= cutoff,
                    VideoMetricsModel.window_type == "24h",
                    VideoMetricsModel.reward_score.isnot(None),
                )
                .group_by(RecipeModel.hook_type)
                .order_by(desc("avg_score"))
                .limit(5)
            )
            .mappings()
            .all()
        )

        return [r["hook_type"] for r in results if r["hook_type"]]

    def _get_avg_video_duration(self, project_id: UUID, cutoff: datetime) -> float:
        """Get average video duration in seconds."""
        from shorts_engine.db.models import VideoRecipeFeaturesModel

        result = self.session.execute(
            select(func.avg(VideoRecipeFeaturesModel.total_duration_seconds))
            .join(VideoJobModel, VideoRecipeFeaturesModel.video_job_id == VideoJobModel.id)
            .where(
                VideoJobModel.project_id == project_id,
                VideoJobModel.created_at >= cutoff,
                VideoRecipeFeaturesModel.total_duration_seconds.isnot(None),
            )
        ).scalar()

        return float(result) if result else 45.0

    async def generate(
        self,
        project_id: UUID,
        n: int = 5,
        temperature: float = 0.8,
        context: TopicContext | None = None,
    ) -> list[GeneratedTopic]:
        """Generate new topic ideas for a project.

        Args:
            project_id: Project to generate topics for
            n: Number of topics to generate
            temperature: Creativity level (higher = more creative)
            context: Optional pre-built context. If not provided, will build from DB.

        Returns:
            List of generated topic ideas
        """
        if context is None:
            context = self.build_context(project_id)

        logger.info(
            "topic_generation_started",
            project_id=str(project_id),
            n=n,
            provider=self.topic_provider.name,
        )

        topics = await self.topic_provider.generate_topics(
            context=context,
            n=n,
            temperature=temperature,
        )

        # Filter out topics that are too similar to recent ones
        filtered = self._filter_duplicates(topics, context.recent_topics)

        logger.info(
            "topic_generation_completed",
            project_id=str(project_id),
            generated=len(topics),
            after_filter=len(filtered),
        )

        return filtered

    def _filter_duplicates(
        self,
        topics: list[GeneratedTopic],
        recent: list[str],
        similarity_threshold: float = 0.7,
    ) -> list[GeneratedTopic]:
        """Filter out topics too similar to recent ones.

        Uses simple Jaccard similarity on words.
        """
        if not recent:
            return topics

        recent_words_sets = [set(t.lower().split()) for t in recent]

        filtered = []
        for topic in topics:
            topic_words = set(topic.topic.lower().split())

            # Check similarity against all recent topics
            is_duplicate = False
            for recent_set in recent_words_sets:
                if not topic_words or not recent_set:
                    continue
                intersection = len(topic_words & recent_set)
                union = len(topic_words | recent_set)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    logger.debug(
                        "topic_filtered_duplicate",
                        topic=topic.topic,
                        similarity=round(similarity, 2),
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(topic)

        return filtered
