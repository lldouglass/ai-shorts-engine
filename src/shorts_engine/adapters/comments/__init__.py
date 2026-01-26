"""Comments ingestion adapters."""

from shorts_engine.adapters.comments.base import CommentsAdapter
from shorts_engine.adapters.comments.stub import StubCommentsAdapter

__all__ = ["CommentsAdapter", "StubCommentsAdapter"]
