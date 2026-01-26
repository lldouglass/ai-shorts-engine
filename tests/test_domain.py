"""Tests for domain models."""

import pytest
from uuid import UUID

from shorts_engine.domain.models import Video, VideoRequest, Job
from shorts_engine.domain.enums import VideoStatus, JobStatus, JobType


def test_video_request_creation() -> None:
    """Test creating a VideoRequest."""
    request = VideoRequest(
        prompt="Generate a video about cats",
        title="Cat Video",
        duration_seconds=30,
    )

    assert request.prompt == "Generate a video about cats"
    assert request.title == "Cat Video"
    assert request.duration_seconds == 30
    assert request.metadata == {}


def test_video_from_request() -> None:
    """Test creating a Video from a VideoRequest."""
    request = VideoRequest(
        prompt="Generate a video about dogs",
        title="Dog Video",
        description="A cute dog video",
        duration_seconds=45,
    )

    video = Video.from_request(request)

    assert isinstance(video.id, UUID)
    assert video.title == "Dog Video"
    assert video.description == "A cute dog video"
    assert video.prompt == "Generate a video about dogs"
    assert video.duration_seconds == 45
    assert video.status == VideoStatus.PENDING
    assert video.file_path is None


def test_video_from_request_auto_title() -> None:
    """Test Video auto-generates title when not provided."""
    request = VideoRequest(prompt="Test prompt")
    video = Video.from_request(request)

    assert video.title.startswith("Video ")


def test_job_creation() -> None:
    """Test creating a Job."""
    job = Job.create(
        job_type=JobType.GENERATE_VIDEO,
        input_data={"prompt": "test"},
    )

    assert isinstance(job.id, UUID)
    assert job.job_type == JobType.GENERATE_VIDEO
    assert job.status == JobStatus.PENDING
    assert job.input_data == {"prompt": "test"}
    assert job.output_data == {}


def test_video_status_enum() -> None:
    """Test VideoStatus enum values."""
    assert VideoStatus.PENDING == "pending"
    assert VideoStatus.GENERATING == "generating"
    assert VideoStatus.PUBLISHED == "published"
    assert VideoStatus.FAILED == "failed"


def test_job_type_enum() -> None:
    """Test JobType enum values."""
    assert JobType.SMOKE_TEST == "smoke_test"
    assert JobType.GENERATE_VIDEO == "generate_video"
    assert JobType.PUBLISH_VIDEO == "publish_video"
