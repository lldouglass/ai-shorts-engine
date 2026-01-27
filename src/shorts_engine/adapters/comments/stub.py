"""Stub comments adapter for testing."""

import random
from datetime import datetime, timedelta
from uuid import uuid4

from shorts_engine.adapters.comments.base import CommentData, CommentsAdapter
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

SAMPLE_COMMENTS = [
    "This is amazing! 🔥",
    "How did you make this?",
    "Love the edit!",
    "Can you do a tutorial?",
    "Not impressed tbh",
    "First!",
    "Who else is watching in 2024?",
    "This needs more views",
    "Algorithm brought me here",
    "The music choice is perfect",
    "I've watched this 10 times",
    "Share more content like this please!",
    "This is exactly what I needed today",
    "Underrated content 💯",
    "Why isn't this viral yet?",
]

SAMPLE_AUTHORS = [
    "user_fan_2024",
    "content_lover",
    "random_viewer_42",
    "shorts_addict",
    "creative_mind",
    "night_owl_123",
    "casual_scroller",
    "video_enthusiast",
]


class StubCommentsAdapter(CommentsAdapter):
    """Stub adapter that returns simulated comments."""

    def __init__(self, platform: Platform = Platform.YOUTUBE) -> None:
        self._platform = platform

    @property
    def platform(self) -> Platform:
        return self._platform

    async def fetch_comments(
        self,
        platform_video_id: str,
        max_results: int = 100,
        since: datetime | None = None,
    ) -> list[CommentData]:
        """Return simulated comments."""
        logger.info(
            "stub_fetch_comments",
            platform=self.platform,
            platform_video_id=platform_video_id,
            max_results=max_results,
        )

        num_comments = min(random.randint(5, 30), max_results)
        comments = []
        base_time = datetime.now()

        for _ in range(num_comments):
            posted_at = base_time - timedelta(
                hours=random.randint(1, 168),  # Up to 1 week ago
                minutes=random.randint(0, 59),
            )

            # Skip if before 'since' timestamp
            if since and posted_at < since:
                continue

            comments.append(
                CommentData(
                    platform=self.platform,
                    platform_comment_id=f"comment_{uuid4().hex[:12]}",
                    platform_video_id=platform_video_id,
                    author=random.choice(SAMPLE_AUTHORS),
                    text=random.choice(SAMPLE_COMMENTS),
                    likes=random.randint(0, 100),
                    posted_at=posted_at,
                    raw_data={"source": "stub"},
                )
            )

        # Sort by posted_at descending (newest first)
        comments.sort(key=lambda c: c.posted_at or datetime.min, reverse=True)

        return comments

    async def reply_to_comment(
        self,
        platform_video_id: str,
        platform_comment_id: str,
        text: str,
    ) -> CommentData | None:
        """Simulate replying to a comment."""
        logger.info(
            "stub_reply_to_comment",
            platform=self.platform,
            platform_video_id=platform_video_id,
            platform_comment_id=platform_comment_id,
        )

        return CommentData(
            platform=self.platform,
            platform_comment_id=f"reply_{uuid4().hex[:12]}",
            platform_video_id=platform_video_id,
            author="shorts_engine_bot",
            text=text,
            likes=0,
            posted_at=datetime.now(),
            reply_to=platform_comment_id,
            raw_data={"source": "stub", "is_reply": True},
        )

    async def health_check(self) -> bool:
        """Stub adapter is always healthy."""
        return True
