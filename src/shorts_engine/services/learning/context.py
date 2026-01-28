"""Optimization context builder for the learning loop.

Aggregates learnings from past performance to inject into LLM prompts,
enabling the system to learn from what works and avoid what doesn't.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from shorts_engine.db.models import (
    ExperimentModel,
    ProjectModel,
    PublishJobModel,
    RecipeModel,
    VideoJobModel,
    VideoMetricsModel,
)
from shorts_engine.domain.enums import ExperimentStatus
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceInsight:
    """A single insight about what works or doesn't."""

    category: str  # "hook_type", "ending_type", "pacing", "style", etc.
    insight: str  # Human-readable insight
    confidence: float  # 0-1 confidence score
    source: str  # "top_performers", "experiment", "bottom_performers"


@dataclass
class OptimizationContext:
    """Context containing learnings to inject into LLM prompts.

    This is the output of OptimizationContextBuilder and gets passed
    to LearningLoopPlanner to inform content generation.
    """

    project_id: UUID
    project_name: str

    # What works (based on top performers)
    winning_patterns: list[PerformanceInsight] = field(default_factory=list)

    # What to avoid (based on poor performers)
    avoid_patterns: list[PerformanceInsight] = field(default_factory=list)

    # Experiment insights (completed A/B tests)
    experiment_insights: list[PerformanceInsight] = field(default_factory=list)

    # Top performing content summaries
    top_topics: list[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lookback_days: int = 30
    sample_size: int = 0

    def format_for_prompt(self) -> str:
        """Format the context for injection into LLM prompts."""
        sections = []

        if self.winning_patterns:
            winning = "\n".join(
                f"- {p.insight} (confidence: {p.confidence:.0%})" for p in self.winning_patterns
            )
            sections.append(f"## What Works Best for This Audience\n{winning}")

        if self.avoid_patterns:
            avoid = "\n".join(
                f"- {p.insight} (confidence: {p.confidence:.0%})" for p in self.avoid_patterns
            )
            sections.append(f"## Patterns to Avoid\n{avoid}")

        if self.experiment_insights:
            experiments = "\n".join(
                f"- {p.insight} (confidence: {p.confidence:.0%})" for p in self.experiment_insights
            )
            sections.append(f"## Recent Experiment Findings\n{experiments}")

        if self.top_topics:
            topics = "\n".join(f"- {t}" for t in self.top_topics[:5])
            sections.append(f"## Top Performing Topic Themes\n{topics}")

        if not sections:
            return ""

        header = f"# Optimization Context (Based on {self.sample_size} videos over {self.lookback_days} days)\n"
        return header + "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "project_id": str(self.project_id),
            "project_name": self.project_name,
            "winning_patterns": [
                {
                    "category": p.category,
                    "insight": p.insight,
                    "confidence": p.confidence,
                    "source": p.source,
                }
                for p in self.winning_patterns
            ],
            "avoid_patterns": [
                {
                    "category": p.category,
                    "insight": p.insight,
                    "confidence": p.confidence,
                    "source": p.source,
                }
                for p in self.avoid_patterns
            ],
            "experiment_insights": [
                {
                    "category": p.category,
                    "insight": p.insight,
                    "confidence": p.confidence,
                    "source": p.source,
                }
                for p in self.experiment_insights
            ],
            "top_topics": self.top_topics,
            "generated_at": self.generated_at.isoformat(),
            "lookback_days": self.lookback_days,
            "sample_size": self.sample_size,
        }


class OptimizationContextBuilder:
    """Builds optimization context from historical performance data.

    Analyzes:
    1. Top-performing videos to identify winning patterns
    2. Poor-performing videos to identify what to avoid
    3. Completed experiments to extract validated insights
    """

    def __init__(self, session: Session):
        """Initialize the builder.

        Args:
            session: Database session
        """
        self.session = session

    def build(
        self,
        project_id: UUID,
        lookback_days: int = 30,
        top_n: int = 20,
        bottom_n: int = 10,
    ) -> OptimizationContext:
        """Build optimization context for a project.

        Args:
            project_id: Project to build context for
            lookback_days: How far back to look for data
            top_n: Number of top performers to analyze
            bottom_n: Number of bottom performers to analyze

        Returns:
            OptimizationContext with aggregated insights
        """
        project = self.session.get(ProjectModel, project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Count total videos in period
        sample_size = self._count_videos(project_id, cutoff)

        context = OptimizationContext(
            project_id=project_id,
            project_name=project.name,
            lookback_days=lookback_days,
            sample_size=sample_size,
        )

        if sample_size < 5:
            logger.info(
                "insufficient_data_for_context",
                project_id=str(project_id),
                sample_size=sample_size,
            )
            return context

        # Analyze top performers
        context.winning_patterns = self._analyze_top_performers(project_id, cutoff, top_n)

        # Analyze bottom performers
        context.avoid_patterns = self._analyze_bottom_performers(project_id, cutoff, bottom_n)

        # Get experiment insights
        context.experiment_insights = self._get_experiment_insights(project_id)

        # Get top topics
        context.top_topics = self._get_top_topics(project_id, cutoff)

        logger.info(
            "optimization_context_built",
            project_id=str(project_id),
            winning_patterns=len(context.winning_patterns),
            avoid_patterns=len(context.avoid_patterns),
            experiment_insights=len(context.experiment_insights),
            sample_size=sample_size,
        )

        return context

    def _count_videos(self, project_id: UUID, cutoff: datetime) -> int:
        """Count videos with metrics in the lookback period."""
        result = self.session.execute(
            select(func.count(VideoJobModel.id))
            .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
            .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
            .where(
                VideoJobModel.project_id == project_id,
                VideoJobModel.created_at >= cutoff,
                VideoMetricsModel.window_type == "24h",
            )
        ).scalar()
        return result or 0

    def _analyze_top_performers(
        self,
        project_id: UUID,
        cutoff: datetime,
        limit: int,
    ) -> list[PerformanceInsight]:
        """Analyze top-performing videos to identify winning patterns."""
        insights = []

        # Get top videos with their recipe data
        top_videos = (
            self.session.execute(
                select(
                    RecipeModel.hook_type,
                    RecipeModel.ending_type,
                    RecipeModel.scene_count,
                    RecipeModel.narration_wpm_bucket,
                    RecipeModel.caption_density_bucket,
                    RecipeModel.preset,
                    VideoMetricsModel.reward_score,
                )
                .join(VideoJobModel, VideoJobModel.recipe_id == RecipeModel.id)
                .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
                .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
                .where(
                    VideoJobModel.project_id == project_id,
                    VideoJobModel.created_at >= cutoff,
                    VideoMetricsModel.window_type == "24h",
                    VideoMetricsModel.reward_score.isnot(None),
                )
                .order_by(desc(VideoMetricsModel.reward_score))
                .limit(limit)
            )
            .mappings()
            .all()
        )

        if not top_videos:
            return insights

        # Analyze patterns in top performers
        hook_counts: dict[str, int] = {}
        ending_counts: dict[str, int] = {}
        pacing_counts: dict[str, int] = {}
        style_counts: dict[str, int] = {}

        for video in top_videos:
            hook_counts[video["hook_type"]] = hook_counts.get(video["hook_type"], 0) + 1
            ending_counts[video["ending_type"]] = ending_counts.get(video["ending_type"], 0) + 1
            pacing_counts[video["narration_wpm_bucket"]] = (
                pacing_counts.get(video["narration_wpm_bucket"], 0) + 1
            )
            style_counts[video["preset"]] = style_counts.get(video["preset"], 0) + 1

        total = len(top_videos)

        # Find dominant patterns (>40% of top performers)
        for hook, count in hook_counts.items():
            ratio = count / total
            if ratio >= 0.4:
                insights.append(
                    PerformanceInsight(
                        category="hook_type",
                        insight=f"{hook} hooks perform best ({ratio:.0%} of top videos use this)",
                        confidence=ratio,
                        source="top_performers",
                    )
                )

        for ending, count in ending_counts.items():
            ratio = count / total
            if ratio >= 0.4:
                insights.append(
                    PerformanceInsight(
                        category="ending_type",
                        insight=f"{ending} endings drive the best engagement ({ratio:.0%} of top videos)",
                        confidence=ratio,
                        source="top_performers",
                    )
                )

        for pacing, count in pacing_counts.items():
            ratio = count / total
            if ratio >= 0.4:
                insights.append(
                    PerformanceInsight(
                        category="pacing",
                        insight=f"{pacing} narration pacing resonates best with this audience",
                        confidence=ratio,
                        source="top_performers",
                    )
                )

        for style, count in style_counts.items():
            ratio = count / total
            if ratio >= 0.4:
                insights.append(
                    PerformanceInsight(
                        category="style",
                        insight=f"{style} visual style performs consistently well",
                        confidence=ratio,
                        source="top_performers",
                    )
                )

        return insights

    def _analyze_bottom_performers(
        self,
        project_id: UUID,
        cutoff: datetime,
        limit: int,
    ) -> list[PerformanceInsight]:
        """Analyze poor-performing videos to identify patterns to avoid."""
        insights = []

        # Get bottom videos with their recipe data
        bottom_videos = (
            self.session.execute(
                select(
                    RecipeModel.hook_type,
                    RecipeModel.ending_type,
                    RecipeModel.scene_count,
                    RecipeModel.narration_wpm_bucket,
                    RecipeModel.preset,
                    VideoMetricsModel.reward_score,
                )
                .join(VideoJobModel, VideoJobModel.recipe_id == RecipeModel.id)
                .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
                .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
                .where(
                    VideoJobModel.project_id == project_id,
                    VideoJobModel.created_at >= cutoff,
                    VideoMetricsModel.window_type == "24h",
                    VideoMetricsModel.reward_score.isnot(None),
                    VideoMetricsModel.reward_score < 0.3,  # Below 30th percentile
                )
                .order_by(VideoMetricsModel.reward_score)
                .limit(limit)
            )
            .mappings()
            .all()
        )

        if len(bottom_videos) < 3:
            return insights

        # Analyze patterns in bottom performers
        hook_counts: dict[str, int] = {}
        ending_counts: dict[str, int] = {}

        for video in bottom_videos:
            hook_counts[video["hook_type"]] = hook_counts.get(video["hook_type"], 0) + 1
            ending_counts[video["ending_type"]] = ending_counts.get(video["ending_type"], 0) + 1

        total = len(bottom_videos)

        # Find patterns that appear frequently in poor performers
        for hook, count in hook_counts.items():
            ratio = count / total
            if ratio >= 0.5:  # >50% of poor performers
                insights.append(
                    PerformanceInsight(
                        category="hook_type",
                        insight=f"Avoid overusing {hook} hooks - they appear in {ratio:.0%} of underperforming videos",
                        confidence=ratio,
                        source="bottom_performers",
                    )
                )

        for ending, count in ending_counts.items():
            ratio = count / total
            if ratio >= 0.5:
                insights.append(
                    PerformanceInsight(
                        category="ending_type",
                        insight=f"{ending} endings tend to underperform for this audience",
                        confidence=ratio,
                        source="bottom_performers",
                    )
                )

        return insights

    def _get_experiment_insights(self, project_id: UUID) -> list[PerformanceInsight]:
        """Get insights from completed experiments."""
        insights = []

        # Get completed experiments with clear winners
        experiments = (
            self.session.execute(
                select(ExperimentModel).where(
                    ExperimentModel.project_id == project_id,
                    ExperimentModel.status == ExperimentStatus.COMPLETED,
                    ExperimentModel.winner.isnot(None),
                )
            )
            .scalars()
            .all()
        )

        for exp in experiments:
            # Determine what the insight is
            if exp.winner == "variant":
                insight = f"Changing {exp.variable_tested} from {exp.baseline_value} to {exp.variant_value} improves performance"
            else:
                insight = f"Keep using {exp.baseline_value} for {exp.variable_tested} - it outperforms {exp.variant_value}"

            confidence = exp.confidence_level if exp.confidence_level else 0.7

            insights.append(
                PerformanceInsight(
                    category=exp.variable_tested,
                    insight=insight,
                    confidence=confidence,
                    source="experiment",
                )
            )

        return insights

    def _get_top_topics(self, project_id: UUID, cutoff: datetime) -> list[str]:
        """Get topics from top-performing videos."""
        results = (
            self.session.execute(
                select(VideoJobModel.idea)
                .join(PublishJobModel, PublishJobModel.video_job_id == VideoJobModel.id)
                .join(VideoMetricsModel, VideoMetricsModel.publish_job_id == PublishJobModel.id)
                .where(
                    VideoJobModel.project_id == project_id,
                    VideoJobModel.created_at >= cutoff,
                    VideoMetricsModel.window_type == "24h",
                    VideoMetricsModel.reward_score >= 0.7,  # Top 30%
                    VideoJobModel.idea.isnot(None),
                )
                .order_by(desc(VideoMetricsModel.reward_score))
                .limit(10)
            )
            .scalars()
            .all()
        )

        return list(results)
