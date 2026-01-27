"""Unit tests for YouTube publishing."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from shorts_engine.adapters.publisher.base import PublishRequest
from shorts_engine.adapters.publisher.youtube import (
    MAX_DESCRIPTION_LENGTH,
    MAX_TITLE_LENGTH,
    YouTubeAccountState,
    YouTubePublisher,
    YouTubeUploadResult,
    build_dry_run_payload,
)
from shorts_engine.domain.enums import Platform


class TestYouTubeAccountState:
    """Tests for YouTubeAccountState."""

    def test_account_state_defaults(self):
        """Test default values for account state."""
        state = YouTubeAccountState(
            account_id=uuid4(),
            access_token="test_access_token",
            refresh_token="test_refresh_token",
        )

        assert state.access_token == "test_access_token"
        assert state.refresh_token == "test_refresh_token"
        assert state.uploads_today == 0
        assert state.max_uploads_per_day == 50

    def test_account_state_with_limits(self):
        """Test account state with custom limits."""
        state = YouTubeAccountState(
            account_id=uuid4(),
            access_token="test",
            refresh_token="test",
            uploads_today=10,
            max_uploads_per_day=25,
        )

        assert state.uploads_today == 10
        assert state.max_uploads_per_day == 25


class TestYouTubePublisher:
    """Tests for YouTubePublisher."""

    @pytest.fixture
    def publisher(self):
        """Create a test publisher in dry-run mode."""
        state = YouTubeAccountState(
            account_id=uuid4(),
            access_token="test_access_token",
            refresh_token="test_refresh_token",
        )
        return YouTubePublisher(account_state=state, dry_run=True)

    def test_platform_property(self, publisher):
        """Test platform property returns YOUTUBE."""
        assert publisher.platform == Platform.YOUTUBE

    def test_dry_run_mode(self, publisher):
        """Test dry run flag is set."""
        assert publisher.dry_run is True

    def test_validate_shorts_requirements_valid(self, publisher):
        """Test validation passes for valid request."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="Test Short",
            description="A test description",
            tags=["test", "short"],
        )

        warnings = publisher._validate_shorts_requirements(request)
        assert len(warnings) == 0

    def test_validate_shorts_requirements_long_title(self, publisher):
        """Test validation warns on long title."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="A" * (MAX_TITLE_LENGTH + 10),
        )

        warnings = publisher._validate_shorts_requirements(request)
        assert any("Title will be truncated" in w for w in warnings)

    def test_validate_shorts_requirements_long_description(self, publisher):
        """Test validation warns on long description."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="Test",
            description="A" * (MAX_DESCRIPTION_LENGTH + 100),
        )

        warnings = publisher._validate_shorts_requirements(request)
        assert any("Description will be truncated" in w for w in warnings)

    def test_build_video_metadata_basic(self, publisher):
        """Test building basic video metadata."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="My Test Video",
            description="This is a test",
            visibility="public",
        )

        metadata = publisher._build_video_metadata(request)

        assert metadata["snippet"]["title"] == "My Test Video"
        assert "#Shorts" in metadata["snippet"]["description"]
        assert metadata["status"]["privacyStatus"] == "public"

    def test_build_video_metadata_with_tags(self, publisher):
        """Test building metadata with tags."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="Test",
            tags=["test", "shorts", "video"],
        )

        metadata = publisher._build_video_metadata(request)

        assert "tags" in metadata["snippet"]
        assert metadata["snippet"]["tags"] == ["test", "shorts", "video"]

    def test_build_video_metadata_scheduled(self, publisher):
        """Test building metadata for scheduled video."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="Test",
            visibility="public",
        )

        scheduled_time = datetime.now(UTC) + timedelta(hours=24)
        metadata = publisher._build_video_metadata(request, scheduled_time)

        # Scheduled videos must be private until publish time
        assert metadata["status"]["privacyStatus"] == "private"
        assert "publishAt" in metadata["status"]

    def test_build_video_metadata_adds_shorts_hashtag(self, publisher):
        """Test that #Shorts is added to description."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="Test",
            description="My video about cats",
        )

        metadata = publisher._build_video_metadata(request)

        assert "#Shorts" in metadata["snippet"]["description"]

    def test_build_video_metadata_preserves_existing_shorts_hashtag(self, publisher):
        """Test that existing #Shorts hashtag is preserved."""
        request = PublishRequest(
            video_path=Path("/tmp/test.mp4"),
            title="Test",
            description="My #Shorts video",
        )

        metadata = publisher._build_video_metadata(request)

        # Should not add duplicate
        assert metadata["snippet"]["description"].count("#Shorts") == 1

    def test_check_rate_limit_allowed(self, publisher):
        """Test rate limit check when uploads allowed."""
        publisher.account_state.uploads_today = 5
        publisher.account_state.max_uploads_per_day = 50

        # Should not raise
        publisher._check_rate_limit()

    def test_check_rate_limit_exceeded(self, publisher):
        """Test rate limit check when limit exceeded."""
        from datetime import datetime, timedelta

        publisher.account_state.uploads_today = 50
        publisher.account_state.max_uploads_per_day = 50
        # Set reset time to tomorrow so it doesn't auto-reset
        tomorrow = datetime.now(UTC) + timedelta(days=1)
        publisher.account_state.uploads_reset_at = tomorrow

        with pytest.raises(ValueError, match="Daily upload limit reached"):
            publisher._check_rate_limit()

    def test_check_rate_limit_resets_new_day(self, publisher):
        """Test rate limit resets on new day."""
        # Set reset time to yesterday
        yesterday = datetime.now(UTC) - timedelta(days=1)
        publisher.account_state.uploads_today = 50
        publisher.account_state.uploads_reset_at = yesterday

        # Should reset counter and not raise
        publisher._check_rate_limit()
        assert publisher.account_state.uploads_today == 0


class TestYouTubeUploadResult:
    """Tests for YouTubeUploadResult."""

    def test_success_result(self):
        """Test successful result creation."""
        result = YouTubeUploadResult(
            success=True,
            video_id="abc123",
            url="https://youtube.com/shorts/abc123",
            visibility="public",
        )

        assert result.success is True
        assert result.video_id == "abc123"
        assert result.url == "https://youtube.com/shorts/abc123"
        assert result.forced_private is False

    def test_failure_result(self):
        """Test failed result creation."""
        result = YouTubeUploadResult(
            success=False,
            error_message="Upload failed: quota exceeded",
        )

        assert result.success is False
        assert result.video_id is None
        assert "quota exceeded" in result.error_message

    def test_forced_private_result(self):
        """Test result when video was forced to private."""
        result = YouTubeUploadResult(
            success=True,
            video_id="abc123",
            url="https://youtube.com/shorts/abc123",
            visibility="private",
            forced_private=True,
        )

        assert result.success is True
        assert result.forced_private is True
        assert result.visibility == "private"


class TestBuildDryRunPayload:
    """Tests for build_dry_run_payload helper."""

    def test_basic_payload(self, tmp_path):
        """Test building basic dry run payload."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video data")

        payload = build_dry_run_payload(
            video_path=video_path,
            title="Test Video",
            description="Test description",
        )

        assert payload["dry_run"] is True
        assert payload["file_exists"] is True
        assert payload["metadata"]["snippet"]["title"] == "Test Video"
        assert "would_upload_to" in payload

    def test_payload_with_schedule(self, tmp_path):
        """Test dry run payload with schedule."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video data")

        scheduled_time = "2024-12-25T10:00:00Z"
        payload = build_dry_run_payload(
            video_path=video_path,
            title="Scheduled Video",
            scheduled_time=scheduled_time,
        )

        assert payload["metadata"]["status"]["privacyStatus"] == "private"
        assert "publishAt" in payload["metadata"]["status"]

    def test_payload_with_tags(self, tmp_path):
        """Test dry run payload with tags."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video data")

        payload = build_dry_run_payload(
            video_path=video_path,
            title="Tagged Video",
            tags=["tag1", "tag2", "tag3"],
        )

        assert payload["metadata"]["snippet"]["tags"] == ["tag1", "tag2", "tag3"]


class TestEncryption:
    """Tests for token encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly."""
        from shorts_engine.services.encryption import decrypt_token, encrypt_token

        original = "my_secret_token_12345"
        encrypted = encrypt_token(original)

        # Encrypted should be different from original
        assert encrypted != original

        # Decrypted should match original
        decrypted = decrypt_token(encrypted)
        assert decrypted == original

    def test_encrypt_empty_fails(self):
        """Test that encrypting empty string fails."""
        from shorts_engine.services.encryption import EncryptionError, encrypt_token

        with pytest.raises(EncryptionError, match="Cannot encrypt empty token"):
            encrypt_token("")

    def test_decrypt_empty_fails(self):
        """Test that decrypting empty string fails."""
        from shorts_engine.services.encryption import EncryptionError, decrypt_token

        with pytest.raises(EncryptionError, match="Cannot decrypt empty token"):
            decrypt_token("")

    def test_decrypt_invalid_fails(self):
        """Test that decrypting invalid data fails."""
        from shorts_engine.services.encryption import EncryptionError, decrypt_token

        with pytest.raises(EncryptionError):
            decrypt_token("not_valid_encrypted_data")


class TestPublishJobModel:
    """Tests for PublishJobModel creation."""

    def test_model_creation(self):
        """Test creating a publish job model."""
        from shorts_engine.db.models import PublishJobModel

        job = PublishJobModel(
            id=uuid4(),
            video_job_id=uuid4(),
            account_id=uuid4(),
            platform="youtube",
            status="pending",
            title="Test Video",
            description="Test description",
            tags=["test", "video"],
            visibility="public",
            dry_run=False,
            forced_private=False,  # Explicit value since server_default only applies on insert
        )

        assert job.platform == "youtube"
        assert job.status == "pending"
        assert job.tags == ["test", "video"]
        assert job.forced_private is False
