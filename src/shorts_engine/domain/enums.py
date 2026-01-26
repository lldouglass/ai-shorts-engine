"""Domain enumerations."""

from enum import StrEnum


class VideoStatus(StrEnum):
    """Status of a video in the pipeline."""

    PENDING = "pending"
    GENERATING = "generating"
    RENDERING = "rendering"
    READY = "ready"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class Platform(StrEnum):
    """Supported publishing platforms."""

    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"


class JobStatus(StrEnum):
    """Status of a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    """Types of background jobs."""

    SMOKE_TEST = "smoke_test"
    GENERATE_VIDEO = "generate_video"
    RENDER_VIDEO = "render_video"
    PUBLISH_VIDEO = "publish_video"
    INGEST_ANALYTICS = "ingest_analytics"
    INGEST_COMMENTS = "ingest_comments"
    ITERATE_STRATEGY = "iterate_strategy"


class Sentiment(StrEnum):
    """Sentiment classification for comments."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
