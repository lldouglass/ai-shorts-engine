"""Instagram Publisher Adapter using Meta Graph API.

Handles video uploads for Instagram Reels via the Content Publishing API.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from shorts_engine.adapters.publisher.base import (
    PublishRequest,
    PublishResponse,
    PublisherAdapter,
)
from shorts_engine.adapters.publisher.instagram_oauth import (
    InstagramOAuthError,
    refresh_instagram_token,
)
from shorts_engine.domain.enums import Platform

logger = logging.getLogger(__name__)

# Meta Graph API endpoints
GRAPH_API_URL = "https://graph.facebook.com/v18.0"

# Instagram Reels requirements
MAX_REELS_DURATION = 90  # seconds (can be up to 15 minutes for some accounts)
MIN_REELS_DURATION = 3  # seconds
MAX_CAPTION_LENGTH = 2200
MAX_HASHTAGS = 30
ASPECT_RATIO_REELS = "9:16"

# Import account state from domain to avoid circular imports
from shorts_engine.domain.account_state import InstagramAccountState


@dataclass
class InstagramUploadResult:
    """Result from an Instagram upload."""

    success: bool
    media_id: str | None = None
    permalink: str | None = None
    error_message: str | None = None
    api_response: dict[str, Any] | None = None


class InstagramPublisher(PublisherAdapter):
    """Instagram Reels publisher using Meta Graph API.

    Features:
    - Video upload via Content Publishing API
    - Container-based async publishing (upload -> poll -> publish)
    - Automatic token refresh
    - Rate limiting (max posts per day)
    - Dry-run mode for testing

    Publishing Flow:
    1. POST /{ig-user-id}/media - Create media container with video_url
    2. GET /{container-id}?fields=status_code - Poll until FINISHED
    3. POST /{ig-user-id}/media_publish - Publish the container
    """

    def __init__(
        self,
        account_state: InstagramAccountState | None = None,
        dry_run: bool = False,
    ):
        """Initialize the Instagram publisher.

        Args:
            account_state: Account credentials and state.
            dry_run: If True, don't actually upload, just log what would happen.
        """
        self.account_state = account_state
        self.dry_run = dry_run
        self._client: httpx.AsyncClient | None = None

    @property
    def platform(self) -> Platform:
        return Platform.INSTAGRAM

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300)  # 5 min timeout for uploads
        return self._client

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token.

        Raises:
            InstagramOAuthError: If token cannot be obtained.
        """
        if not self.account_state:
            raise InstagramOAuthError("No account credentials configured")

        # Check if token needs refresh (7 days before expiry)
        now = datetime.now(timezone.utc)
        if (
            self.account_state.token_expires_at
            and self.account_state.token_expires_at > now + timedelta(days=7)
        ):
            # Token still valid
            return self.account_state.access_token

        # Refresh token
        logger.info("Refreshing Instagram access token")
        try:
            refresh_result = refresh_instagram_token(self.account_state.access_token)
            self.account_state.access_token = refresh_result["access_token"]
            self.account_state.token_expires_at = now + timedelta(
                seconds=refresh_result.get("expires_in", 5184000)  # 60 days default
            )
            return self.account_state.access_token
        except Exception as e:
            raise InstagramOAuthError(f"Failed to refresh access token: {e}")

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

    def _validate_reels_requirements(self, request: PublishRequest) -> list[str]:
        """Validate video meets Instagram Reels requirements.

        Returns:
            List of validation warnings (not errors).
        """
        warnings = []

        # Caption length
        if request.description and len(request.description) > MAX_CAPTION_LENGTH:
            warnings.append(f"Caption will be truncated (max {MAX_CAPTION_LENGTH} chars)")

        # Hashtag count
        if request.description:
            hashtag_count = request.description.count("#")
            if hashtag_count > MAX_HASHTAGS:
                warnings.append(f"Too many hashtags ({hashtag_count}), max is {MAX_HASHTAGS}")

        return warnings

    def _build_caption(self, request: PublishRequest) -> str:
        """Build Instagram caption from request.

        Args:
            request: Publish request.

        Returns:
            Caption string.
        """
        parts = []

        if request.title:
            parts.append(request.title)

        if request.description:
            parts.append(request.description)

        # Add hashtags from tags
        if request.tags:
            hashtags = " ".join(f"#{tag.replace(' ', '')}" for tag in request.tags[:10])
            parts.append(hashtags)

        caption = "\n\n".join(parts)
        return caption[:MAX_CAPTION_LENGTH]

    async def publish(self, request: PublishRequest) -> PublishResponse:
        """Publish a video to Instagram Reels.

        Args:
            request: Publish request with video and metadata.

        Returns:
            PublishResponse with media ID and URL or error.
        """
        try:
            # Validate requirements
            warnings = self._validate_reels_requirements(request)
            for warning in warnings:
                logger.warning(f"Instagram Reels: {warning}")

            # Check rate limit
            self._check_rate_limit()

            # Build caption
            caption = self._build_caption(request)

            # Dry run mode
            if self.dry_run:
                logger.info("DRY RUN: Would upload video to Instagram Reels")
                logger.info(f"DRY RUN: Caption: {caption[:100]}...")
                logger.info(f"DRY RUN: Video path: {request.video_path}")

                return PublishResponse(
                    success=True,
                    platform=Platform.INSTAGRAM,
                    platform_video_id="DRY_RUN_MEDIA_ID",
                    url="https://www.instagram.com/reel/DRY_RUN_MEDIA_ID",
                    metadata={
                        "dry_run": True,
                        "would_upload": str(request.video_path),
                        "caption": caption,
                    },
                )

            # Get access token
            access_token = await self._get_access_token()

            # Video must be accessible via public URL for Instagram API
            # The video_path should be a URL or we need to upload to a hosting service
            video_url = self._get_video_url(request.video_path)
            if not video_url:
                return PublishResponse(
                    success=False,
                    platform=Platform.INSTAGRAM,
                    error_message="Video must be accessible via public URL for Instagram publishing. "
                    "Upload video to a CDN or enable public URL generation.",
                )

            # Upload video
            result = await self._upload_reel(
                video_url=video_url,
                caption=caption,
                access_token=access_token,
            )

            if result.success:
                # Increment post counter
                if self.account_state:
                    self.account_state.posts_today += 1

                return PublishResponse(
                    success=True,
                    platform=Platform.INSTAGRAM,
                    platform_video_id=result.media_id,
                    url=result.permalink,
                    metadata={"api_response": result.api_response},
                )
            else:
                return PublishResponse(
                    success=False,
                    platform=Platform.INSTAGRAM,
                    error_message=result.error_message,
                    metadata={"api_response": result.api_response},
                )

        except Exception as e:
            logger.exception(f"Instagram upload failed: {e}")
            return PublishResponse(
                success=False,
                platform=Platform.INSTAGRAM,
                error_message=str(e),
            )

    def _get_video_url(self, video_path: Path) -> str | None:
        """Get public URL for video file.

        For Instagram API, videos must be accessible via HTTPS URL.
        Local files need to be uploaded to a CDN first.

        Args:
            video_path: Path to video file (may be a URL string).

        Returns:
            Public URL or None if not available.
        """
        path_str = str(video_path)

        # If it's already a URL, return it
        if path_str.startswith("http://") or path_str.startswith("https://"):
            return path_str

        # Local file - would need CDN upload
        # For now, return None to indicate video URL is needed
        logger.warning(
            f"Local file path provided: {video_path}. "
            "Instagram API requires public video URL. "
            "Video must be uploaded to CDN first."
        )
        return None

    async def _upload_reel(
        self,
        video_url: str,
        caption: str,
        access_token: str,
    ) -> InstagramUploadResult:
        """Upload a Reel to Instagram using Content Publishing API.

        Args:
            video_url: Public URL of the video.
            caption: Caption text.
            access_token: Valid access token.

        Returns:
            InstagramUploadResult with media ID or error.
        """
        if not self.account_state:
            return InstagramUploadResult(
                success=False,
                error_message="No account state configured",
            )

        client = await self._get_client()
        ig_user_id = self.account_state.instagram_account_id

        # Step 1: Create media container
        container_response = await client.post(
            f"{GRAPH_API_URL}/{ig_user_id}/media",
            params={
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption,
                "access_token": access_token,
            },
        )

        if container_response.status_code != 200:
            error_data = self._safe_json(container_response)
            error_msg = self._extract_error(error_data, container_response.text)
            return InstagramUploadResult(
                success=False,
                error_message=f"Failed to create media container: {error_msg}",
                api_response=error_data,
            )

        container_data = container_response.json()
        container_id = container_data.get("id")

        if not container_id:
            return InstagramUploadResult(
                success=False,
                error_message="No container ID in response",
                api_response=container_data,
            )

        logger.info(f"Instagram media container created: {container_id}")

        # Step 2: Poll for container status
        max_attempts = 60  # Max 5 minutes with 5s interval
        for attempt in range(max_attempts):
            status_response = await client.get(
                f"{GRAPH_API_URL}/{container_id}",
                params={
                    "fields": "status_code,status",
                    "access_token": access_token,
                },
            )

            if status_response.status_code != 200:
                await asyncio.sleep(5)
                continue

            status_data = status_response.json()
            status_code = status_data.get("status_code")

            if status_code == "FINISHED":
                logger.info(f"Instagram media container ready: {container_id}")
                break
            elif status_code == "ERROR":
                return InstagramUploadResult(
                    success=False,
                    error_message=f"Media processing failed: {status_data.get('status')}",
                    api_response=status_data,
                )
            elif status_code == "IN_PROGRESS":
                logger.debug(f"Container processing in progress (attempt {attempt + 1})")
                await asyncio.sleep(5)
            else:
                await asyncio.sleep(5)
        else:
            return InstagramUploadResult(
                success=False,
                error_message="Media processing timed out",
            )

        # Step 3: Publish the container
        publish_response = await client.post(
            f"{GRAPH_API_URL}/{ig_user_id}/media_publish",
            params={
                "creation_id": container_id,
                "access_token": access_token,
            },
        )

        if publish_response.status_code != 200:
            error_data = self._safe_json(publish_response)
            error_msg = self._extract_error(error_data, publish_response.text)
            return InstagramUploadResult(
                success=False,
                error_message=f"Failed to publish media: {error_msg}",
                api_response=error_data,
            )

        publish_data = publish_response.json()
        media_id = publish_data.get("id")

        # Get permalink
        permalink = await self._get_permalink(media_id, access_token)

        logger.info(f"Instagram Reel published: {media_id} ({permalink})")

        return InstagramUploadResult(
            success=True,
            media_id=media_id,
            permalink=permalink,
            api_response=publish_data,
        )

    async def _get_permalink(self, media_id: str, access_token: str) -> str | None:
        """Get the permalink for a published media.

        Args:
            media_id: The published media ID.
            access_token: Valid access token.

        Returns:
            Permalink URL or None.
        """
        client = await self._get_client()

        response = await client.get(
            f"{GRAPH_API_URL}/{media_id}",
            params={
                "fields": "permalink",
                "access_token": access_token,
            },
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("permalink")

        return None

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
                "id": platform_video_id,
                "media_type": "REELS",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dry_run": True,
            }

        access_token = await self._get_access_token()
        client = await self._get_client()

        response = await client.get(
            f"{GRAPH_API_URL}/{platform_video_id}",
            params={
                "fields": "id,media_type,media_url,permalink,timestamp,like_count,comments_count",
                "access_token": access_token,
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get media status: {response.text}")

        return response.json()

    async def delete_video(self, platform_video_id: str) -> bool:
        """Delete a video from Instagram.

        Note: Instagram Graph API doesn't support media deletion.
        Media can only be deleted through the Instagram app.
        """
        logger.warning(
            f"Instagram API does not support media deletion. "
            f"Media {platform_video_id} must be deleted through Instagram app."
        )
        return False

    async def health_check(self) -> bool:
        """Check if Instagram API is accessible and authenticated."""
        if self.dry_run:
            return True

        try:
            access_token = await self._get_access_token()
            client = await self._get_client()

            if not self.account_state:
                return False

            response = await client.get(
                f"{GRAPH_API_URL}/{self.account_state.instagram_account_id}",
                params={
                    "fields": "id,username",
                    "access_token": access_token,
                },
            )

            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Instagram health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def build_instagram_dry_run_payload(
    video_url: str,
    title: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a payload showing what would be uploaded in dry-run mode.

    Useful for testing and validation without making actual API calls.
    """
    from shorts_engine.adapters.publisher.base import PublishRequest

    publisher = InstagramPublisher(dry_run=True)

    request = PublishRequest(
        video_path=Path(video_url),
        title=title,
        description=description,
        tags=tags,
    )

    caption = publisher._build_caption(request)

    return {
        "video_url": video_url,
        "caption": caption,
        "media_type": "REELS",
        "would_upload_to": "Instagram Reels",
        "dry_run": True,
    }
