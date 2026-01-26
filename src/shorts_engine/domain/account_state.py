"""Account state dataclasses for platform publishers.

These are separated from the publisher adapters to avoid circular imports
with the services module.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


# YouTube
DEFAULT_YOUTUBE_MAX_UPLOADS_PER_DAY = 50


@dataclass
class YouTubeAccountState:
    """State for a YouTube account."""

    account_id: UUID
    access_token: str
    refresh_token: str
    token_expires_at: datetime | None = None
    uploads_today: int = 0
    uploads_reset_at: datetime | None = None
    max_uploads_per_day: int = DEFAULT_YOUTUBE_MAX_UPLOADS_PER_DAY


# Instagram
DEFAULT_INSTAGRAM_MAX_POSTS_PER_DAY = 25


@dataclass
class InstagramAccountState:
    """State for an Instagram account."""

    account_id: UUID
    access_token: str
    instagram_account_id: str  # IG Business Account ID
    token_expires_at: datetime | None = None
    posts_today: int = 0
    posts_reset_at: datetime | None = None
    max_posts_per_day: int = DEFAULT_INSTAGRAM_MAX_POSTS_PER_DAY


# TikTok
DEFAULT_TIKTOK_MAX_POSTS_PER_DAY = 50


@dataclass
class TikTokAccountState:
    """State for a TikTok account."""

    account_id: UUID
    access_token: str
    refresh_token: str
    open_id: str
    token_expires_at: datetime | None = None
    posts_today: int = 0
    posts_reset_at: datetime | None = None
    max_posts_per_day: int = DEFAULT_TIKTOK_MAX_POSTS_PER_DAY
    has_direct_post: bool = False  # Whether Direct Post capability is approved
