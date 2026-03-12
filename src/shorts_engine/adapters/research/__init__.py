"""Content research adapters for trend discovery."""

from shorts_engine.adapters.research.base import (
    ContentCategory,
    ResearchProvider,
    ResearchResult,
    TrendSignal,
    TrendSource,
)
from shorts_engine.adapters.research.tiktok import TikTokResearchProvider
from shorts_engine.adapters.research.youtube_trends import YouTubeResearchProvider

__all__ = [
    "ContentCategory",
    "ResearchProvider",
    "ResearchResult",
    "TrendSignal",
    "TrendSource",
    "TikTokResearchProvider",
    "YouTubeResearchProvider",
]
