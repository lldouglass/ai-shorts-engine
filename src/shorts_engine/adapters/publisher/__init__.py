"""Video publishing adapters."""

from shorts_engine.adapters.publisher.base import PublisherAdapter
from shorts_engine.adapters.publisher.stub import StubPublisherAdapter

__all__ = ["PublisherAdapter", "StubPublisherAdapter"]
