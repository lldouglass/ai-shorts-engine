"""Comments ingestion adapters."""

from shorts_engine.adapters.comments.base import CommentData, CommentsAdapter
from shorts_engine.adapters.comments.instagram import InstagramCommentsAdapter
from shorts_engine.adapters.comments.stub import StubCommentsAdapter
from shorts_engine.adapters.comments.tiktok import TikTokCommentsAdapter
from shorts_engine.adapters.comments.youtube import YouTubeCommentsAdapter

__all__ = [
    "CommentsAdapter",
    "CommentData",
    "StubCommentsAdapter",
    "YouTubeCommentsAdapter",
    "InstagramCommentsAdapter",
    "TikTokCommentsAdapter",
]
