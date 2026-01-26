"""Analytics ingestion adapters."""

from shorts_engine.adapters.analytics.base import AnalyticsAdapter, MetricsSnapshot
from shorts_engine.adapters.analytics.stub import StubAnalyticsAdapter
from shorts_engine.adapters.analytics.youtube import YouTubeAnalyticsAdapter
from shorts_engine.adapters.analytics.instagram import InstagramAnalyticsAdapter
from shorts_engine.adapters.analytics.tiktok import TikTokAnalyticsAdapter

__all__ = [
    "AnalyticsAdapter",
    "MetricsSnapshot",
    "StubAnalyticsAdapter",
    "YouTubeAnalyticsAdapter",
    "InstagramAnalyticsAdapter",
    "TikTokAnalyticsAdapter",
]
