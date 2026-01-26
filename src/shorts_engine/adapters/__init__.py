"""Adapters for external services."""

from shorts_engine.adapters.analytics.base import AnalyticsAdapter
from shorts_engine.adapters.comments.base import CommentsAdapter
from shorts_engine.adapters.publisher.base import PublisherAdapter
from shorts_engine.adapters.renderer.base import RendererProvider
from shorts_engine.adapters.video_gen.base import VideoGenProvider

__all__ = [
    "AnalyticsAdapter",
    "CommentsAdapter",
    "PublisherAdapter",
    "RendererProvider",
    "VideoGenProvider",
]
