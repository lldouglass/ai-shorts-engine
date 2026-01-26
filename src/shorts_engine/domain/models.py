"""Domain models - pure Python classes independent of database."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from shorts_engine.domain.enums import JobStatus, JobType, Platform, Sentiment, VideoStatus


@dataclass
class VideoRequest:
    """Request to generate a new video."""

    prompt: str
    title: str | None = None
    description: str | None = None
    duration_seconds: int = 60
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Video:
    """A generated video in the pipeline."""

    id: UUID
    title: str
    description: str | None
    prompt: str
    duration_seconds: int
    status: VideoStatus
    file_path: str | None = None
    file_size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_request(cls, request: VideoRequest) -> "Video":
        """Create a Video from a VideoRequest."""
        return cls(
            id=uuid4(),
            title=request.title or f"Video {uuid4().hex[:8]}",
            description=request.description,
            prompt=request.prompt,
            duration_seconds=request.duration_seconds,
            status=VideoStatus.PENDING,
            metadata=request.metadata,
        )


@dataclass
class PublishResult:
    """Result of publishing a video to a platform."""

    id: UUID
    video_id: UUID
    platform: Platform
    platform_video_id: str | None = None
    url: str | None = None
    status: VideoStatus = VideoStatus.PENDING
    error_message: str | None = None
    published_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a published video."""

    id: UUID
    publish_result_id: UUID
    views: int = 0
    likes: int = 0
    comments_count: int = 0
    shares: int = 0
    watch_time_seconds: int = 0
    avg_view_duration_seconds: float | None = None
    engagement_rate: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime | None = None

    @property
    def calculated_engagement_rate(self) -> float | None:
        """Calculate engagement rate from available metrics."""
        if self.views == 0:
            return None
        return (self.likes + self.comments_count + self.shares) / self.views


@dataclass
class Comment:
    """A comment on a published video."""

    id: UUID
    publish_result_id: UUID
    platform_comment_id: str
    text: str
    author: str | None = None
    sentiment: Sentiment | None = None
    likes: int = 0
    posted_at: datetime | None = None
    fetched_at: datetime | None = None


@dataclass
class Job:
    """A background job record."""

    id: UUID
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    celery_task_id: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    @classmethod
    def create(cls, job_type: JobType, input_data: dict[str, Any] | None = None) -> "Job":
        """Create a new job."""
        return cls(
            id=uuid4(),
            job_type=job_type,
            input_data=input_data or {},
        )
