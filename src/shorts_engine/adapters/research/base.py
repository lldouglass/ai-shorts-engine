"""Base interface for content research providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any


class TrendSource(str, Enum):
    """Where the trend signal came from."""
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    GOOGLE_TRENDS = "google_trends"
    MANUAL = "manual"


class ContentCategory(str, Enum):
    """Broad content categories for classification."""
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    COMEDY = "comedy"
    STORYTELLING = "storytelling"
    HORROR_DARK = "horror_dark"
    SCIFI_FANTASY = "scifi_fantasy"
    ANIME = "anime"
    MOTIVATION = "motivation"
    NEWS_CURRENT = "news_current"
    TECH = "tech"
    GAMING = "gaming"
    LIFESTYLE = "lifestyle"
    OTHER = "other"


@dataclass
class TrendSignal:
    """A single trend signal from any source.

    Represents a trending topic, video format, or content pattern
    discovered from scraping platforms.
    """
    title: str
    source: TrendSource
    url: str | None = None

    # Engagement metrics (platform-specific, normalized where possible)
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0

    # Trend metadata
    hashtags: list[str] = field(default_factory=list)
    category: ContentCategory = ContentCategory.OTHER
    description: str | None = None
    creator: str | None = None
    creator_followers: int | None = None

    # Timing
    published_at: datetime | None = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # How fast is it growing? (views per hour since publish, if calculable)
    velocity: float | None = None

    # Raw data from the source for debugging
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate (likes + comments + shares) / views."""
        if self.views == 0:
            return 0.0
        return (self.likes + self.comments + self.shares) / self.views

    @property
    def virality_score(self) -> float:
        """Heuristic virality score 0-1 based on available metrics.

        Weights velocity heavily since fast growth = viral potential.
        """
        score = 0.0

        # View volume (log scale, max ~0.3)
        if self.views > 0:
            import math
            score += min(0.3, math.log10(max(self.views, 1)) / 25)

        # Engagement rate (max ~0.3)
        score += min(0.3, self.engagement_rate * 3)

        # Velocity bonus (max ~0.4)
        if self.velocity and self.velocity > 0:
            import math
            score += min(0.4, math.log10(max(self.velocity, 1)) / 15)

        return min(1.0, score)


@dataclass
class ResearchResult:
    """Aggregated research results from a provider."""
    source: TrendSource
    signals: list[TrendSignal]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.signals) > 0


class ResearchProvider(ABC):
    """Abstract base class for content research providers.

    Each provider scrapes/queries a specific platform for trending
    content signals that can inform video topic generation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    @abstractmethod
    def source(self) -> TrendSource:
        """Which platform this provider scrapes."""
        ...

    @abstractmethod
    async def fetch_trends(
        self,
        categories: list[str] | None = None,
        limit: int = 50,
    ) -> ResearchResult:
        """Fetch current trending content signals.

        Args:
            categories: Optional list of categories/niches to focus on.
            limit: Maximum number of signals to return.

        Returns:
            ResearchResult with trend signals.
        """
        ...

    async def fetch_competitor_videos(
        self,
        channel_ids: list[str] | None = None,
        limit: int = 50,
    ) -> ResearchResult:
        """Fetch top-performing videos from competitor channels.

        Not all providers support this. Default returns empty result.
        """
        return ResearchResult(
            source=self.source,
            signals=[],
            error="Competitor analysis not supported by this provider",
        )

    async def health_check(self) -> bool:
        """Check if the provider is operational."""
        return True
