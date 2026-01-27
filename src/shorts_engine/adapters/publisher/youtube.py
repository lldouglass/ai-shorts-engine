"""YouTube Publisher Adapter using YouTube Data API v3.

Handles video uploads, scheduling, and status tracking for YouTube Shorts.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from shorts_engine.adapters.publisher.base import (
    PublisherAdapter,
    PublishRequest,
    PublishResponse,
)
from shorts_engine.adapters.publisher.youtube_oauth import (
    OAuthError,
    refresh_access_token,
)
from shorts_engine.domain.enums import Platform

logger = logging.getLogger(__name__)

# YouTube API endpoints
YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# YouTube Shorts requirements
MAX_SHORTS_DURATION = 60  # seconds
MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 5000
MAX_TAGS_TOTAL_LENGTH = 500

# Rate limiting constant moved to domain.account_state


@dataclass
class YouTubeUploadResult:
    """Result from a YouTube upload."""

    success: bool
    video_id: str | None = None
    url: str | None = None
    visibility: str | None = None
    forced_private: bool = False
    error_message: str | None = None
    api_response: dict[str, Any] | None = None


# Import account state from domain to avoid circular imports
from shorts_engine.domain.account_state import YouTubeAccountState  # noqa: E402


class YouTubePublisher(PublisherAdapter):
    """YouTube Shorts publisher using YouTube Data API.

    Features:
    - Video upload with resumable uploads for large files
    - Scheduled publishing (publishAt)
    - Automatic token refresh
    - Rate limiting (max uploads per day)
    - Dry-run mode for testing
    """

    def __init__(
        self,
        account_state: YouTubeAccountState | None = None,
        dry_run: bool = False,
    ):
        """Initialize the YouTube publisher.

        Args:
            account_state: Account credentials and state.
            dry_run: If True, don't actually upload, just log what would happen.
        """
        self.account_state = account_state
        self.dry_run = dry_run
        self._client: httpx.Client | None = None

    @property
    def platform(self) -> Platform:
        return Platform.YOUTUBE

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=300)  # 5 min timeout for uploads
        return self._client

    def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token.

        Raises:
            OAuthError: If token cannot be obtained.
        """
        if not self.account_state:
            raise OAuthError("No account credentials configured")

        # Check if token needs refresh
        now = datetime.now(UTC)
        if (
            self.account_state.token_expires_at
            and self.account_state.token_expires_at > now + timedelta(minutes=5)
        ):
            # Token still valid
            return self.account_state.access_token

        # Refresh token
        logger.info("Refreshing YouTube access token")
        try:
            refresh_result = refresh_access_token(self.account_state.refresh_token)
            self.account_state.access_token = refresh_result["access_token"]
            self.account_state.token_expires_at = now + timedelta(
                seconds=refresh_result["expires_in"]
            )
            return self.account_state.access_token
        except Exception as e:
            raise OAuthError(f"Failed to refresh access token: {e}")

    def _check_rate_limit(self) -> None:
        """Check if we've exceeded the daily upload limit.

        Raises:
            ValueError: If rate limit exceeded.
        """
        if not self.account_state:
            return

        now = datetime.now(UTC)

        # Reset counter if it's a new day
        if self.account_state.uploads_reset_at is None or self.account_state.uploads_reset_at < now:
            self.account_state.uploads_today = 0
            # Reset at midnight UTC
            tomorrow = now.date() + timedelta(days=1)
            self.account_state.uploads_reset_at = datetime(
                tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=UTC
            )

        if self.account_state.uploads_today >= self.account_state.max_uploads_per_day:
            raise ValueError(
                f"Daily upload limit reached ({self.account_state.max_uploads_per_day} uploads). "
                f"Resets at {self.account_state.uploads_reset_at.isoformat()}"
            )

    def _validate_shorts_requirements(self, request: PublishRequest) -> list[str]:
        """Validate video meets YouTube Shorts requirements.

        Returns:
            List of validation warnings (not errors).
        """
        warnings = []

        # Title length
        if len(request.title) > MAX_TITLE_LENGTH:
            warnings.append(f"Title will be truncated (max {MAX_TITLE_LENGTH} chars)")

        # Description length
        if request.description and len(request.description) > MAX_DESCRIPTION_LENGTH:
            warnings.append(f"Description will be truncated (max {MAX_DESCRIPTION_LENGTH} chars)")

        # Tags total length
        if request.tags:
            total_tag_length = sum(len(t) for t in request.tags)
            if total_tag_length > MAX_TAGS_TOTAL_LENGTH:
                warnings.append(f"Tags may be truncated (total max {MAX_TAGS_TOTAL_LENGTH} chars)")

        return warnings

    def _build_video_metadata(
        self,
        request: PublishRequest,
        scheduled_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Build video metadata for YouTube API.

        Args:
            request: Publish request.
            scheduled_time: Optional scheduled publish time.

        Returns:
            Video resource metadata dict.
        """
        # Determine visibility and scheduling
        visibility = request.visibility
        publish_at = None

        if scheduled_time:
            # YouTube requires private visibility for scheduled videos
            visibility = "private"
            publish_at = scheduled_time.isoformat()

        # Build snippet
        snippet: dict[str, Any] = {
            "title": request.title[:MAX_TITLE_LENGTH],
            "categoryId": "22",  # People & Blogs (good default for Shorts)
        }

        if request.description:
            # Add #Shorts to description if not present
            description = request.description[:MAX_DESCRIPTION_LENGTH]
            if "#Shorts" not in description and "#shorts" not in description:
                description = f"{description}\n\n#Shorts"
            snippet["description"] = description
        else:
            snippet["description"] = "#Shorts"

        if request.tags:
            # Filter and validate tags
            valid_tags = [t for t in request.tags if t and len(t) <= 100]
            total_length = 0
            final_tags = []
            for tag in valid_tags:
                if total_length + len(tag) <= MAX_TAGS_TOTAL_LENGTH:
                    final_tags.append(tag)
                    total_length += len(tag)
            snippet["tags"] = final_tags

        # Build status
        status = {
            "privacyStatus": visibility,
            "selfDeclaredMadeForKids": False,
        }

        if publish_at:
            status["publishAt"] = publish_at

        return {
            "snippet": snippet,
            "status": status,
        }

    async def publish(self, request: PublishRequest) -> PublishResponse:
        """Publish a video to YouTube.

        Args:
            request: Publish request with video and metadata.

        Returns:
            PublishResponse with video ID and URL or error.
        """
        try:
            # Validate requirements
            warnings = self._validate_shorts_requirements(request)
            for warning in warnings:
                logger.warning(f"YouTube Shorts: {warning}")

            # Check rate limit
            self._check_rate_limit()

            # Parse scheduled time if provided
            scheduled_time = None
            if request.scheduled_time:
                scheduled_time = datetime.fromisoformat(
                    request.scheduled_time.replace("Z", "+00:00")
                )

            # Build metadata
            metadata = self._build_video_metadata(request, scheduled_time)

            # Dry run mode
            if self.dry_run:
                logger.info("DRY RUN: Would upload video to YouTube")
                logger.info(f"DRY RUN: Metadata: {metadata}")
                logger.info(f"DRY RUN: Video path: {request.video_path}")

                return PublishResponse(
                    success=True,
                    platform=Platform.YOUTUBE,
                    platform_video_id="DRY_RUN_VIDEO_ID",
                    url="https://youtube.com/shorts/DRY_RUN_VIDEO_ID",
                    metadata={
                        "dry_run": True,
                        "would_upload": str(request.video_path),
                        "metadata": metadata,
                    },
                )

            # Get access token
            access_token = self._get_access_token()

            # Upload video
            result = self._upload_video(
                video_path=request.video_path,
                metadata=metadata,
                access_token=access_token,
            )

            if result.success:
                # Increment upload counter
                if self.account_state:
                    self.account_state.uploads_today += 1

                return PublishResponse(
                    success=True,
                    platform=Platform.YOUTUBE,
                    platform_video_id=result.video_id,
                    url=result.url,
                    metadata={
                        "visibility": result.visibility,
                        "forced_private": result.forced_private,
                        "api_response": result.api_response,
                    },
                )
            else:
                return PublishResponse(
                    success=False,
                    platform=Platform.YOUTUBE,
                    error_message=result.error_message,
                    metadata={"api_response": result.api_response},
                )

        except Exception as e:
            logger.exception(f"YouTube upload failed: {e}")
            return PublishResponse(
                success=False,
                platform=Platform.YOUTUBE,
                error_message=str(e),
            )

    def _upload_video(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        access_token: str,
    ) -> YouTubeUploadResult:
        """Upload video file to YouTube.

        Uses simple upload for small files, resumable for larger ones.
        """
        file_size = video_path.stat().st_size

        # Read video file
        with open(video_path, "rb") as f:
            video_data = f.read()

        # Use simple upload for files under 5MB, otherwise resumable
        if file_size < 5 * 1024 * 1024:
            return self._simple_upload(video_data, metadata, access_token)
        else:
            return self._resumable_upload(video_path, metadata, access_token)

    def _simple_upload(
        self,
        video_data: bytes,
        metadata: dict[str, Any],
        access_token: str,
    ) -> YouTubeUploadResult:
        """Simple single-request upload for small videos."""
        import json

        # Prepare multipart request
        boundary = "shorts_engine_upload_boundary"

        # Build multipart body
        body_parts = []

        # Metadata part
        body_parts.append(f"--{boundary}")
        body_parts.append("Content-Type: application/json; charset=UTF-8")
        body_parts.append("")
        body_parts.append(json.dumps(metadata))

        # Video part
        body_parts.append(f"--{boundary}")
        body_parts.append("Content-Type: video/mp4")
        body_parts.append("Content-Transfer-Encoding: binary")
        body_parts.append("")

        # Combine text parts
        text_body = "\r\n".join(body_parts) + "\r\n"
        final_boundary = f"\r\n--{boundary}--"

        # Build full body
        full_body = text_body.encode() + video_data + final_boundary.encode()

        # Upload
        response = self.client.post(
            YOUTUBE_UPLOAD_URL,
            params={
                "uploadType": "multipart",
                "part": "snippet,status",
            },
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
            },
            content=full_body,
        )

        return self._parse_upload_response(response)

    def _resumable_upload(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        access_token: str,
    ) -> YouTubeUploadResult:
        """Resumable upload for larger videos."""

        file_size = video_path.stat().st_size

        # Step 1: Initialize upload
        init_response = self.client.post(
            YOUTUBE_UPLOAD_URL,
            params={
                "uploadType": "resumable",
                "part": "snippet,status",
            },
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(file_size),
                "X-Upload-Content-Type": "video/mp4",
            },
            json=metadata,
        )

        if init_response.status_code != 200:
            return YouTubeUploadResult(
                success=False,
                error_message=f"Failed to initialize upload: {init_response.text}",
                api_response=self._safe_json(init_response),
            )

        upload_url = init_response.headers.get("Location")
        if not upload_url:
            return YouTubeUploadResult(
                success=False,
                error_message="No upload URL in response",
            )

        # Step 2: Upload video data
        with open(video_path, "rb") as f:
            upload_response = self.client.put(
                upload_url,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(file_size),
                },
                content=f,
            )

        return self._parse_upload_response(upload_response)

    def _parse_upload_response(self, response: httpx.Response) -> YouTubeUploadResult:
        """Parse YouTube upload API response."""
        if response.status_code in (200, 201):
            data = response.json()
            video_id = data.get("id")
            visibility = data.get("status", {}).get("privacyStatus")

            # Check if upload was forced to private
            upload_status = data.get("status", {}).get("uploadStatus")
            forced_private = (
                visibility == "private"
                and upload_status in ("uploaded", "processed")
                and data.get("status", {}).get("privacyStatus") != "private"
            )

            return YouTubeUploadResult(
                success=True,
                video_id=video_id,
                url=f"https://youtube.com/shorts/{video_id}",
                visibility=visibility,
                forced_private=forced_private,
                api_response=data,
            )
        else:
            error_data = self._safe_json(response)
            error_msg = "Upload failed"

            if error_data:
                error_info = error_data.get("error", {})
                error_msg = error_info.get("message", response.text)

                # Check for specific errors
                errors = error_info.get("errors", [])
                for err in errors:
                    reason = err.get("reason", "")
                    if reason == "quotaExceeded":
                        error_msg = "YouTube API quota exceeded. Try again tomorrow."
                    elif reason == "uploadLimitExceeded":
                        error_msg = "YouTube upload limit exceeded for this account."
                    elif reason == "videoNotFound":
                        error_msg = "Video processing failed on YouTube's side."

            return YouTubeUploadResult(
                success=False,
                error_message=error_msg,
                api_response=error_data,
            )

    def _safe_json(self, response: httpx.Response) -> dict[str, Any] | None:
        """Safely parse JSON response."""
        try:
            return response.json()  # type: ignore[no-any-return]
        except Exception:
            return None

    async def get_video_status(self, platform_video_id: str) -> dict[str, Any]:
        """Get the current status of an uploaded video."""
        if self.dry_run:
            return {
                "id": platform_video_id,
                "status": {"uploadStatus": "processed", "privacyStatus": "public"},
                "dry_run": True,
            }

        access_token = self._get_access_token()

        response = self.client.get(
            YOUTUBE_VIDEOS_URL,
            params={
                "id": platform_video_id,
                "part": "status,snippet,statistics",
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get video status: {response.text}")

        data = response.json()
        items = data.get("items", [])

        if not items:
            raise ValueError(f"Video not found: {platform_video_id}")

        return items[0]  # type: ignore[no-any-return]

    async def delete_video(self, platform_video_id: str) -> bool:
        """Delete a video from YouTube."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would delete video {platform_video_id}")
            return True

        access_token = self._get_access_token()

        response = self.client.delete(
            YOUTUBE_VIDEOS_URL,
            params={"id": platform_video_id},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        return response.status_code == 204

    async def health_check(self) -> bool:
        """Check if YouTube API is accessible and authenticated."""
        if self.dry_run:
            return True

        try:
            access_token = self._get_access_token()

            # Try to get channel info
            response = self.client.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "snippet", "mine": "true"},
                headers={"Authorization": f"Bearer {access_token}"},
            )

            return response.status_code == 200
        except Exception as e:
            logger.warning(f"YouTube health check failed: {e}")
            return False

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if self._client:
            self._client.close()


def build_dry_run_payload(
    video_path: Path,
    title: str,
    description: str | None = None,
    tags: list[str] | None = None,
    scheduled_time: str | None = None,
    visibility: str = "public",
) -> dict[str, Any]:
    """Build a payload showing what would be uploaded in dry-run mode.

    Useful for testing and validation without making actual API calls.
    """
    publisher = YouTubePublisher(dry_run=True)

    request = PublishRequest(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        scheduled_time=scheduled_time,
        visibility=visibility,
    )

    # Parse scheduled time
    parsed_scheduled = None
    if scheduled_time:
        parsed_scheduled = datetime.fromisoformat(scheduled_time.replace("Z", "+00:00"))

    metadata = publisher._build_video_metadata(request, parsed_scheduled)

    return {
        "video_path": str(video_path),
        "file_exists": video_path.exists() if video_path else False,
        "metadata": metadata,
        "would_upload_to": "YouTube Shorts",
        "dry_run": True,
    }
