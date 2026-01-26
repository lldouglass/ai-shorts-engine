"""TikTok Publisher Adapter using Content Posting API.

Handles video uploads via TikTok's Direct Post API with fallback to
NEEDS_MANUAL_PUBLISH status for accounts without Direct Post approval.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx

from shorts_engine.adapters.publisher.base import (
    PublishRequest,
    PublishResponse,
    PublisherAdapter,
)
from shorts_engine.adapters.publisher.tiktok_oauth import (
    TikTokOAuthError,
    check_direct_post_capability,
    refresh_tiktok_token,
)
from shorts_engine.domain.enums import Platform, PublishStatus

logger = logging.getLogger(__name__)

# TikTok Content Posting API endpoints
TIKTOK_POST_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
TIKTOK_POST_STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
TIKTOK_UPLOAD_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"

# TikTok video requirements
MAX_VIDEO_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB
MIN_VIDEO_DURATION = 1  # seconds
MAX_VIDEO_DURATION = 600  # 10 minutes
MAX_TITLE_LENGTH = 150
MAX_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB

# Import account state from domain to avoid circular imports
from shorts_engine.domain.account_state import TikTokAccountState


@dataclass
class TikTokUploadResult:
    """Result from a TikTok upload."""

    success: bool
    publish_id: str | None = None
    share_url: str | None = None
    status: str | None = None
    error_message: str | None = None
    api_response: dict[str, Any] | None = None
    needs_manual_publish: bool = False
    manual_publish_path: str | None = None


class TikTokPublisher(PublisherAdapter):
    """TikTok publisher using Content Posting API.

    Features:
    - Direct Post (if approved): Upload and publish directly
    - Fallback to NEEDS_MANUAL_PUBLISH with video path stored
    - Automatic token refresh
    - Rate limiting (max posts per day)
    - Dry-run mode for testing

    Publishing Flow (Direct Post):
    1. POST /v2/post/publish/video/init/ - Initialize upload
    2. PUT upload_url - Upload video chunks
    3. POST /v2/post/publish/status/fetch/ - Poll for completion
    """

    def __init__(
        self,
        account_state: TikTokAccountState | None = None,
        dry_run: bool = False,
    ):
        """Initialize the TikTok publisher.

        Args:
            account_state: Account credentials and state.
            dry_run: If True, don't actually upload, just log what would happen.
        """
        self.account_state = account_state
        self.dry_run = dry_run
        self._client: httpx.AsyncClient | None = None

    @property
    def platform(self) -> Platform:
        return Platform.TIKTOK

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=600)  # 10 min timeout for uploads
        return self._client

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token.

        Raises:
            TikTokOAuthError: If token cannot be obtained.
        """
        if not self.account_state:
            raise TikTokOAuthError("No account credentials configured")

        # Check if token needs refresh (1 hour before expiry)
        now = datetime.now(timezone.utc)
        if (
            self.account_state.token_expires_at
            and self.account_state.token_expires_at > now + timedelta(hours=1)
        ):
            # Token still valid
            return self.account_state.access_token

        # Refresh token
        logger.info("Refreshing TikTok access token")
        try:
            refresh_result = refresh_tiktok_token(self.account_state.refresh_token)
            self.account_state.access_token = refresh_result["access_token"]
            self.account_state.refresh_token = refresh_result.get(
                "refresh_token", self.account_state.refresh_token
            )
            self.account_state.token_expires_at = now + timedelta(
                seconds=refresh_result.get("expires_in", 86400)
            )
            return self.account_state.access_token
        except Exception as e:
            raise TikTokOAuthError(f"Failed to refresh access token: {e}")

    def _check_rate_limit(self) -> None:
        """Check if we've exceeded the daily post limit.

        Raises:
            ValueError: If rate limit exceeded.
        """
        if not self.account_state:
            return

        now = datetime.now(timezone.utc)

        # Reset counter if it's a new day
        if (
            self.account_state.posts_reset_at is None
            or self.account_state.posts_reset_at < now
        ):
            self.account_state.posts_today = 0
            # Reset at midnight UTC
            tomorrow = now.date() + timedelta(days=1)
            self.account_state.posts_reset_at = datetime(
                tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc
            )

        if self.account_state.posts_today >= self.account_state.max_posts_per_day:
            raise ValueError(
                f"Daily post limit reached ({self.account_state.max_posts_per_day} posts). "
                f"Resets at {self.account_state.posts_reset_at.isoformat()}"
            )

    def _validate_video_requirements(self, request: PublishRequest) -> list[str]:
        """Validate video meets TikTok requirements.

        Returns:
            List of validation warnings.
        """
        warnings = []

        # Check file exists
        if not request.video_path.exists():
            warnings.append(f"Video file not found: {request.video_path}")

        # Check file size
        if request.video_path.exists():
            file_size = request.video_path.stat().st_size
            if file_size > MAX_VIDEO_SIZE:
                warnings.append(f"Video too large ({file_size / 1e9:.1f} GB), max is 4 GB")

        # Title length
        if request.title and len(request.title) > MAX_TITLE_LENGTH:
            warnings.append(f"Title will be truncated (max {MAX_TITLE_LENGTH} chars)")

        return warnings

    def _build_title(self, request: PublishRequest) -> str:
        """Build TikTok title/caption from request.

        Args:
            request: Publish request.

        Returns:
            Title string.
        """
        parts = []

        if request.title:
            parts.append(request.title)

        if request.description:
            parts.append(request.description[:100])  # Truncate description

        # Add hashtags
        if request.tags:
            hashtags = " ".join(f"#{tag.replace(' ', '')}" for tag in request.tags[:5])
            parts.append(hashtags)

        title = " | ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
        return title[:MAX_TITLE_LENGTH]

    def _generate_share_url(self, video_path: Path, title: str) -> str:
        """Generate a TikTok Share Intent URL for manual publishing.

        Args:
            video_path: Path to the video file.
            title: Title/caption for the video.

        Returns:
            Share Intent URL.
        """
        # TikTok Share Intent for mobile app
        params = {
            "title": title,
        }
        return f"tiktok://share?{urlencode(params)}"

    async def publish(self, request: PublishRequest) -> PublishResponse:
        """Publish a video to TikTok.

        Args:
            request: Publish request with video and metadata.

        Returns:
            PublishResponse with video ID and URL, or NEEDS_MANUAL_PUBLISH status.
        """
        try:
            # Validate requirements
            warnings = self._validate_video_requirements(request)
            for warning in warnings:
                logger.warning(f"TikTok: {warning}")

            # Check rate limit
            self._check_rate_limit()

            # Build title
            title = self._build_title(request)

            # Dry run mode
            if self.dry_run:
                logger.info("DRY RUN: Would upload video to TikTok")
                logger.info(f"DRY RUN: Title: {title}")
                logger.info(f"DRY RUN: Video path: {request.video_path}")

                return PublishResponse(
                    success=True,
                    platform=Platform.TIKTOK,
                    platform_video_id="DRY_RUN_VIDEO_ID",
                    url="https://www.tiktok.com/@user/video/DRY_RUN_VIDEO_ID",
                    metadata={
                        "dry_run": True,
                        "would_upload": str(request.video_path),
                        "title": title,
                    },
                )

            # Check Direct Post capability
            if not self.account_state:
                return PublishResponse(
                    success=False,
                    platform=Platform.TIKTOK,
                    error_message="No account state configured",
                )

            access_token = await self._get_access_token()

            # Re-check Direct Post capability
            has_direct_post = self.account_state.has_direct_post
            if not has_direct_post:
                has_direct_post = check_direct_post_capability(
                    access_token, self.account_state.open_id
                )
                self.account_state.has_direct_post = has_direct_post

            if not has_direct_post:
                # Fallback to NEEDS_MANUAL_PUBLISH
                logger.warning(
                    "TikTok Direct Post not available. Video stored for manual publishing."
                )

                share_url = self._generate_share_url(request.video_path, title)

                return PublishResponse(
                    success=True,  # Not a failure, just needs manual action
                    platform=Platform.TIKTOK,
                    metadata={
                        "status": PublishStatus.NEEDS_MANUAL_PUBLISH,
                        "needs_manual_publish": True,
                        "manual_publish_path": str(request.video_path),
                        "share_url": share_url,
                        "title": title,
                        "message": "Video stored for manual publishing. Use the TikTok app to upload.",
                    },
                )

            # Direct Post is available - upload video
            result = await self._upload_video(
                video_path=request.video_path,
                title=title,
                access_token=access_token,
            )

            if result.success:
                # Increment post counter
                self.account_state.posts_today += 1

                return PublishResponse(
                    success=True,
                    platform=Platform.TIKTOK,
                    platform_video_id=result.publish_id,
                    url=f"https://www.tiktok.com/@user/video/{result.publish_id}",
                    metadata={
                        "api_response": result.api_response,
                        "share_url": result.share_url,
                    },
                )
            else:
                return PublishResponse(
                    success=False,
                    platform=Platform.TIKTOK,
                    error_message=result.error_message,
                    metadata={"api_response": result.api_response},
                )

        except Exception as e:
            logger.exception(f"TikTok upload failed: {e}")
            return PublishResponse(
                success=False,
                platform=Platform.TIKTOK,
                error_message=str(e),
            )

    async def _upload_video(
        self,
        video_path: Path,
        title: str,
        access_token: str,
    ) -> TikTokUploadResult:
        """Upload a video using TikTok Content Posting API.

        Args:
            video_path: Path to the video file.
            title: Video title/caption.
            access_token: Valid access token.

        Returns:
            TikTokUploadResult with publish ID or error.
        """
        client = await self._get_client()

        # Get file info
        file_size = video_path.stat().st_size
        chunk_size = min(MAX_CHUNK_SIZE, file_size)
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        # Step 1: Initialize upload
        init_response = await client.post(
            TIKTOK_POST_INIT_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={
                "post_info": {
                    "title": title,
                    "privacy_level": "SELF_ONLY",  # Start as private, user can change later
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": file_size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": total_chunks,
                },
            },
        )

        if init_response.status_code != 200:
            error_data = self._safe_json(init_response)
            error_msg = self._extract_error(error_data, init_response.text)
            return TikTokUploadResult(
                success=False,
                error_message=f"Failed to initialize upload: {error_msg}",
                api_response=error_data,
            )

        init_data = init_response.json()

        if init_data.get("error", {}).get("code") != "ok":
            error_msg = init_data.get("error", {}).get("message", "Unknown error")
            return TikTokUploadResult(
                success=False,
                error_message=f"Upload init failed: {error_msg}",
                api_response=init_data,
            )

        upload_url = init_data.get("data", {}).get("upload_url")
        publish_id = init_data.get("data", {}).get("publish_id")

        if not upload_url or not publish_id:
            return TikTokUploadResult(
                success=False,
                error_message="No upload URL or publish ID in response",
                api_response=init_data,
            )

        logger.info(f"TikTok upload initialized: {publish_id}")

        # Step 2: Upload video chunks
        with open(video_path, "rb") as f:
            for chunk_num in range(total_chunks):
                chunk_data = f.read(chunk_size)
                start_byte = chunk_num * chunk_size
                end_byte = start_byte + len(chunk_data) - 1

                upload_response = await client.put(
                    upload_url,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Range": f"bytes {start_byte}-{end_byte}/{file_size}",
                    },
                    content=chunk_data,
                )

                if upload_response.status_code not in (200, 201, 206):
                    return TikTokUploadResult(
                        success=False,
                        error_message=f"Chunk upload failed: {upload_response.text}",
                    )

                logger.debug(f"Uploaded chunk {chunk_num + 1}/{total_chunks}")

        # Step 3: Poll for completion
        max_attempts = 60  # Max 5 minutes
        for attempt in range(max_attempts):
            status_response = await client.post(
                TIKTOK_POST_STATUS_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={"publish_id": publish_id},
            )

            if status_response.status_code != 200:
                await asyncio.sleep(5)
                continue

            status_data = status_response.json()

            if status_data.get("error", {}).get("code") != "ok":
                await asyncio.sleep(5)
                continue

            status = status_data.get("data", {}).get("status")

            if status == "PUBLISH_COMPLETE":
                logger.info(f"TikTok video published: {publish_id}")
                return TikTokUploadResult(
                    success=True,
                    publish_id=publish_id,
                    status="published",
                    api_response=status_data,
                )
            elif status == "FAILED":
                fail_reason = status_data.get("data", {}).get("fail_reason", "Unknown")
                return TikTokUploadResult(
                    success=False,
                    error_message=f"Video processing failed: {fail_reason}",
                    api_response=status_data,
                )
            elif status in ("PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD", "SENDING_TO_USER_INBOX"):
                logger.debug(f"TikTok processing: {status} (attempt {attempt + 1})")
                await asyncio.sleep(5)
            else:
                await asyncio.sleep(5)

        return TikTokUploadResult(
            success=False,
            error_message="Video processing timed out",
        )

    def _extract_error(self, data: dict | None, fallback: str) -> str:
        """Extract error message from API response."""
        if data:
            error = data.get("error", {})
            return error.get("message", fallback)
        return fallback

    def _safe_json(self, response: httpx.Response) -> dict[str, Any] | None:
        """Safely parse JSON response."""
        try:
            return response.json()
        except Exception:
            return None

    async def get_video_status(self, platform_video_id: str) -> dict[str, Any]:
        """Get the current status of a published video."""
        if self.dry_run:
            return {
                "publish_id": platform_video_id,
                "status": "PUBLISH_COMPLETE",
                "dry_run": True,
            }

        access_token = await self._get_access_token()
        client = await self._get_client()

        response = await client.post(
            TIKTOK_POST_STATUS_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={"publish_id": platform_video_id},
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get video status: {response.text}")

        return response.json()

    async def delete_video(self, platform_video_id: str) -> bool:
        """Delete a video from TikTok.

        Note: TikTok API doesn't support programmatic video deletion.
        Videos must be deleted through the TikTok app.
        """
        logger.warning(
            f"TikTok API does not support video deletion. "
            f"Video {platform_video_id} must be deleted through TikTok app."
        )
        return False

    async def health_check(self) -> bool:
        """Check if TikTok API is accessible and authenticated."""
        if self.dry_run:
            return True

        try:
            access_token = await self._get_access_token()

            # Try to get user info
            client = await self._get_client()
            response = await client.get(
                "https://open.tiktokapis.com/v2/user/info/",
                params={"fields": "open_id,display_name"},
                headers={"Authorization": f"Bearer {access_token}"},
            )

            return response.status_code == 200
        except Exception as e:
            logger.warning(f"TikTok health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def build_tiktok_dry_run_payload(
    video_path: str,
    title: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a payload showing what would be uploaded in dry-run mode."""
    from shorts_engine.adapters.publisher.base import PublishRequest

    publisher = TikTokPublisher(dry_run=True)

    request = PublishRequest(
        video_path=Path(video_path),
        title=title,
        description=description,
        tags=tags,
    )

    built_title = publisher._build_title(request)

    return {
        "video_path": str(video_path),
        "title": built_title,
        "privacy_level": "SELF_ONLY",
        "would_upload_to": "TikTok",
        "dry_run": True,
    }
