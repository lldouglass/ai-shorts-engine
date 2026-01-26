"""Analytics ingestion adapters."""

from shorts_engine.adapters.analytics.base import AnalyticsAdapter
from shorts_engine.adapters.analytics.stub import StubAnalyticsAdapter

__all__ = ["AnalyticsAdapter", "StubAnalyticsAdapter"]
