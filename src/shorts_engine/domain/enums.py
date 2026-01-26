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


class HookType(StrEnum):
    """Types of video hooks."""

    QUESTION = "question"  # Opens with a compelling question
    STATEMENT = "statement"  # Bold statement/claim
    VISUAL = "visual"  # Eye-catching visual hook
    STORY = "story"  # Mini-story opening
    CONTRAST = "contrast"  # Before/after or comparison
    MYSTERY = "mystery"  # Creates curiosity gap


class EndingType(StrEnum):
    """Types of video endings."""

    CLIFFHANGER = "cliffhanger"  # Leaves viewer wanting more
    RESOLVE = "resolve"  # Complete resolution/conclusion
    CTA = "cta"  # Call to action
    LOOP = "loop"  # Loops back to beginning


class NarrationWPMBucket(StrEnum):
    """Narration speed buckets (words per minute)."""

    SLOW = "slow"  # < 120 WPM
    MEDIUM = "medium"  # 120-160 WPM
    FAST = "fast"  # > 160 WPM

    @classmethod
    def from_wpm(cls, wpm: float | None) -> "NarrationWPMBucket":
        """Get bucket from actual WPM value."""
        if wpm is None:
            return cls.MEDIUM
        if wpm < 120:
            return cls.SLOW
        elif wpm <= 160:
            return cls.MEDIUM
        else:
            return cls.FAST


class CaptionDensityBucket(StrEnum):
    """Caption density buckets."""

    SPARSE = "sparse"  # < 0.3 captions per second
    MEDIUM = "medium"  # 0.3-0.6 captions per second
    DENSE = "dense"  # > 0.6 captions per second

    @classmethod
    def from_density(cls, density: float | None) -> "CaptionDensityBucket":
        """Get bucket from actual density value."""
        if density is None:
            return cls.MEDIUM
        if density < 0.3:
            return cls.SPARSE
        elif density <= 0.6:
            return cls.MEDIUM
        else:
            return cls.DENSE


class GenerationMode(StrEnum):
    """How a video job was generated."""

    EXPLOIT = "exploit"  # Based on top-performing recipe
    EXPLORE = "explore"  # A/B test variant
    MANUAL = "manual"  # User-created


class ExperimentStatus(StrEnum):
    """Status of an A/B experiment."""

    RUNNING = "running"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"


class BatchStatus(StrEnum):
    """Status of a planned batch."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
