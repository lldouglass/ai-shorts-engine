"""Celery tasks for video publishing pipeline.

Handles publishing videos to YouTube, Instagram, and TikTok with multi-account support.
"""

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
from celery import shared_task

from shorts_engine.adapters.publisher.youtube import (
    YouTubePublisher,
    YouTubeAccountState,
    build_dry_run_payload,
)
from shorts_engine.adapters.publisher.youtube_oauth import OAuthError
from shorts_engine.adapters.publisher.instagram import (
    InstagramPublisher,
    build_instagram_dry_run_payload,
)
from shorts_engine.adapters.publisher.instagram_oauth import InstagramOAuthError
from shorts_engine.adapters.publisher.tiktok import (
    TikTokPublisher,
    build_tiktok_dry_run_payload,
)
from shorts_engine.adapters.publisher.tiktok_oauth import TikTokOAuthError
from shorts_engine.config import settings
from shorts_engine.db.models import (
    AssetModel,
    PlatformAccountModel,
    PublishJobModel,
    VideoJobModel,
)
from shorts_engine.db.session import get_session_context
from shorts_engine.domain.enums import PublishStatus
from shorts_engine.services.accounts import (
    AccountError,
    AccountNotFoundError,
    check_upload_limit,
    get_account_by_id,
    get_account_by_label,
    get_account_state,
    get_instagram_account_state,
    get_tiktok_account_state,
    increment_upload_count,
    mark_account_revoked,
    update_account_tokens,
)
from shorts_engine.services.encryption import decrypt_token

logger = logging.getLogger(__name__)


class PublishError(Exception):
    """Raised when publishing fails."""

    pass


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(httpx.RequestError, httpx.TimeoutException),
    retry_backoff=True,
    retry_backoff_max=600,
)
def publish_to_youtube_task(
    self,
    video_job_id: str,
    account_label: str,
    scheduled_publish_at: str | None = None,
    visibility: str = "public",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish a video to YouTube.

    Args:
        video_job_id: UUID of the video job to publish.
        account_label: Label of the YouTube account to use.
        scheduled_publish_at: Optional ISO 8601 datetime for scheduled publishing.
        visibility: Video visibility (public, private, unlisted).
        dry_run: If True, don't actually upload.

    Returns:
        Dict with publish result.
    """
    logger.info(
        f"Publishing video job {video_job_id} to YouTube account '{account_label}' "
        f"(dry_run={dry_run})"
    )

    with get_session_context() as session:
        # Get video job
        video_job = session.get(VideoJobModel, UUID(video_job_id))
        if not video_job:
            return {"success": False, "error": f"Video job not found: {video_job_id}"}

        # Get account
        try:
            account = get_account_by_label(session, "youtube", account_label)
        except AccountNotFoundError as e:
            return {"success": False, "error": str(e)}

        # Check upload limit
        can_upload, uploads_today, max_uploads = check_upload_limit(session, account.id)
        if not can_upload and not dry_run:
            return {
                "success": False,
                "error": f"Daily upload limit reached ({uploads_today}/{max_uploads}). "
                "Try again tomorrow.",
            }

        # Get final video asset
        final_asset = None
        for asset in video_job.assets:
            if asset.asset_type == "final_video" and asset.status == "ready":
                final_asset = asset
                break

        if not final_asset:
            return {
                "success": False,
                "error": "No final video asset found. Run the render pipeline first.",
            }

        # Get video file
        video_path = None
        if final_asset.file_path:
            video_path = Path(final_asset.file_path)
        elif final_asset.url:
            # Download video from URL
            try:
                video_path = _download_video(final_asset.url)
            except Exception as e:
                return {"success": False, "error": f"Failed to download video: {e}"}

        if not video_path or not video_path.exists():
            return {"success": False, "error": "Video file not found"}

        # Create publish job record
        publish_job = PublishJobModel(
            id=uuid4(),
            video_job_id=video_job.id,
            account_id=account.id,
            platform="youtube",
            status="pending",
            scheduled_publish_at=(
                datetime.fromisoformat(scheduled_publish_at.replace("Z", "+00:00"))
                if scheduled_publish_at
                else None
            ),
            visibility=visibility,
            title=video_job.title,
            description=video_job.description,
            tags=video_job.plan_data.get("tags", []) if video_job.plan_data else None,
            dry_run=dry_run,
        )
        session.add(publish_job)
        session.commit()

        try:
            # Get account state with decrypted tokens
            account_state = get_account_state(session, account.id)

            # Create publisher
            publisher = YouTubePublisher(
                account_state=account_state,
                dry_run=dry_run,
            )

            # Build request
            from shorts_engine.adapters.publisher.base import PublishRequest

            request = PublishRequest(
                video_path=video_path,
                title=video_job.title or "Untitled Short",
                description=video_job.description,
                tags=video_job.plan_data.get("tags", []) if video_job.plan_data else None,
                scheduled_time=scheduled_publish_at,
                visibility=visibility,
            )

            # Dry run - save what would be uploaded
            if dry_run:
                payload = build_dry_run_payload(
                    video_path=video_path,
                    title=request.title,
                    description=request.description,
                    tags=request.tags,
                    scheduled_time=scheduled_publish_at,
                    visibility=visibility,
                )
                publish_job.dry_run_payload = payload
                publish_job.status = "dry_run_complete"
                session.commit()

                return {
                    "success": True,
                    "publish_job_id": str(publish_job.id),
                    "dry_run": True,
                    "payload": payload,
                }

            # Publish
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(publisher.publish(request))
            finally:
                loop.close()

            # Update publish job
            if result.success:
                publish_job.status = "published"
                publish_job.platform_video_id = result.platform_video_id
                publish_job.platform_url = result.url
                publish_job.actual_publish_at = datetime.now(timezone.utc)
                publish_job.api_response = result.metadata

                # Check if forced private
                if result.metadata and result.metadata.get("forced_private"):
                    publish_job.forced_private = True
                    publish_job.visibility = "private"

                # Increment upload count
                increment_upload_count(session, account.id)

                # Update tokens if refreshed
                if publisher.account_state:
                    update_account_tokens(
                        session,
                        account.id,
                        publisher.account_state.access_token,
                        publisher.account_state.token_expires_at,
                    )

                session.commit()

                logger.info(
                    f"Published video to YouTube: {result.url} "
                    f"(video ID: {result.platform_video_id})"
                )

                return {
                    "success": True,
                    "publish_job_id": str(publish_job.id),
                    "platform_video_id": result.platform_video_id,
                    "url": result.url,
                    "forced_private": publish_job.forced_private,
                }
            else:
                publish_job.status = "failed"
                publish_job.error_message = result.error_message
                publish_job.api_response = result.metadata
                session.commit()

                return {
                    "success": False,
                    "publish_job_id": str(publish_job.id),
                    "error": result.error_message,
                }

        except OAuthError as e:
            # Mark account as potentially revoked
            if "invalid_grant" in str(e).lower():
                mark_account_revoked(session, account.id, str(e))

            publish_job.status = "failed"
            publish_job.error_message = str(e)
            session.commit()

            return {
                "success": False,
                "publish_job_id": str(publish_job.id),
                "error": str(e),
            }

        except Exception as e:
            logger.exception(f"Publishing failed: {e}")

            publish_job.status = "failed"
            publish_job.error_message = str(e)
            publish_job.retry_count = (publish_job.retry_count or 0) + 1
            session.commit()

            # Re-raise for Celery retry
            raise


def _download_video(url: str) -> Path:
    """Download video from URL to temp file.

    Args:
        url: Video URL.

    Returns:
        Path to downloaded file.
    """
    logger.info(f"Downloading video from {url}")

    with httpx.Client(timeout=300) as client:
        response = client.get(url)
        response.raise_for_status()

        # Create temp file
        fd, path = tempfile.mkstemp(suffix=".mp4")
        with open(fd, "wb") as f:
            f.write(response.content)

        return Path(path)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(httpx.RequestError, httpx.TimeoutException),
    retry_backoff=True,
    retry_backoff_max=600,
)
def publish_to_instagram_task(
    self,
    video_job_id: str,
    account_label: str,
    scheduled_publish_at: str | None = None,
    visibility: str = "public",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish a video to Instagram Reels.

    Args:
        video_job_id: UUID of the video job to publish.
        account_label: Label of the Instagram account to use.
        scheduled_publish_at: Optional ISO 8601 datetime for scheduled publishing.
        visibility: Video visibility (ignored for Instagram, always public).
        dry_run: If True, don't actually upload.

    Returns:
        Dict with publish result.
    """
    logger.info(
        f"Publishing video job {video_job_id} to Instagram account '{account_label}' "
        f"(dry_run={dry_run})"
    )

    with get_session_context() as session:
        # Get video job
        video_job = session.get(VideoJobModel, UUID(video_job_id))
        if not video_job:
            return {"success": False, "error": f"Video job not found: {video_job_id}"}

        # Get account
        try:
            account = get_account_by_label(session, "instagram", account_label)
        except AccountNotFoundError as e:
            return {"success": False, "error": str(e)}

        # Get final video asset
        final_asset = None
        for asset in video_job.assets:
            if asset.asset_type == "final_video" and asset.status == "ready":
                final_asset = asset
                break

        if not final_asset:
            return {
                "success": False,
                "error": "No final video asset found. Run the render pipeline first.",
            }

        # Get video URL (Instagram requires public URL)
        video_url = final_asset.url
        video_path = None
        if final_asset.file_path:
            video_path = Path(final_asset.file_path)

        # Create publish job record
        publish_job = PublishJobModel(
            id=uuid4(),
            video_job_id=video_job.id,
            account_id=account.id,
            platform="instagram",
            status="pending",
            visibility="public",  # Instagram Reels are always public
            title=video_job.title,
            description=video_job.description,
            tags=video_job.plan_data.get("tags", []) if video_job.plan_data else None,
            dry_run=dry_run,
        )
        session.add(publish_job)
        session.commit()

        try:
            # Get account state
            account_state = get_instagram_account_state(session, account.id)

            # Create publisher
            publisher = InstagramPublisher(
                account_state=account_state,
                dry_run=dry_run,
            )

            # Build request
            from shorts_engine.adapters.publisher.base import PublishRequest

            request = PublishRequest(
                video_path=video_path or Path(video_url or ""),
                title=video_job.title or "Untitled Short",
                description=video_job.description,
                tags=video_job.plan_data.get("tags", []) if video_job.plan_data else None,
                visibility="public",
            )

            # Dry run
            if dry_run:
                payload = build_instagram_dry_run_payload(
                    video_url=video_url or str(video_path),
                    title=request.title,
                    description=request.description,
                    tags=request.tags,
                )
                publish_job.dry_run_payload = payload
                publish_job.status = "dry_run_complete"
                session.commit()

                return {
                    "success": True,
                    "publish_job_id": str(publish_job.id),
                    "dry_run": True,
                    "payload": payload,
                }

            # Publish
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(publisher.publish(request))
            finally:
                loop.close()

            # Update publish job
            if result.success:
                publish_job.status = "published"
                publish_job.platform_video_id = result.platform_video_id
                publish_job.platform_url = result.url
                publish_job.actual_publish_at = datetime.now(timezone.utc)
                publish_job.api_response = result.metadata

                increment_upload_count(session, account.id)
                session.commit()

                logger.info(
                    f"Published video to Instagram: {result.url} "
                    f"(media ID: {result.platform_video_id})"
                )

                return {
                    "success": True,
                    "publish_job_id": str(publish_job.id),
                    "platform_video_id": result.platform_video_id,
                    "url": result.url,
                }
            else:
                publish_job.status = "failed"
                publish_job.error_message = result.error_message
                publish_job.api_response = result.metadata
                session.commit()

                return {
                    "success": False,
                    "publish_job_id": str(publish_job.id),
                    "error": result.error_message,
                }

        except InstagramOAuthError as e:
            if "expired" in str(e).lower():
                mark_account_revoked(session, account.id, str(e))

            publish_job.status = "failed"
            publish_job.error_message = str(e)
            session.commit()

            return {
                "success": False,
                "publish_job_id": str(publish_job.id),
                "error": str(e),
            }

        except Exception as e:
            logger.exception(f"Instagram publishing failed: {e}")

            publish_job.status = "failed"
            publish_job.error_message = str(e)
            publish_job.retry_count = (publish_job.retry_count or 0) + 1
            session.commit()

            raise


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(httpx.RequestError, httpx.TimeoutException),
    retry_backoff=True,
    retry_backoff_max=600,
)
def publish_to_tiktok_task(
    self,
    video_job_id: str,
    account_label: str,
    scheduled_publish_at: str | None = None,
    visibility: str = "public",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish a video to TikTok.

    If Direct Post is not available for the account, sets status to NEEDS_MANUAL_PUBLISH
    and stores the video path for manual publishing.

    Args:
        video_job_id: UUID of the video job to publish.
        account_label: Label of the TikTok account to use.
        scheduled_publish_at: Optional ISO 8601 datetime for scheduled publishing.
        visibility: Video visibility.
        dry_run: If True, don't actually upload.

    Returns:
        Dict with publish result.
    """
    logger.info(
        f"Publishing video job {video_job_id} to TikTok account '{account_label}' "
        f"(dry_run={dry_run})"
    )

    with get_session_context() as session:
        # Get video job
        video_job = session.get(VideoJobModel, UUID(video_job_id))
        if not video_job:
            return {"success": False, "error": f"Video job not found: {video_job_id}"}

        # Get account
        try:
            account = get_account_by_label(session, "tiktok", account_label)
        except AccountNotFoundError as e:
            return {"success": False, "error": str(e)}

        # Get final video asset
        final_asset = None
        for asset in video_job.assets:
            if asset.asset_type == "final_video" and asset.status == "ready":
                final_asset = asset
                break

        if not final_asset:
            return {
                "success": False,
                "error": "No final video asset found. Run the render pipeline first.",
            }

        # Get video path
        video_path = None
        if final_asset.file_path:
            video_path = Path(final_asset.file_path)
        elif final_asset.url:
            try:
                video_path = _download_video(final_asset.url)
            except Exception as e:
                return {"success": False, "error": f"Failed to download video: {e}"}

        if not video_path or not video_path.exists():
            return {"success": False, "error": "Video file not found"}

        # Create publish job record
        publish_job = PublishJobModel(
            id=uuid4(),
            video_job_id=video_job.id,
            account_id=account.id,
            platform="tiktok",
            status="pending",
            visibility=visibility,
            title=video_job.title,
            description=video_job.description,
            tags=video_job.plan_data.get("tags", []) if video_job.plan_data else None,
            dry_run=dry_run,
        )
        session.add(publish_job)
        session.commit()

        try:
            # Get account state
            account_state = get_tiktok_account_state(session, account.id)

            # Create publisher
            publisher = TikTokPublisher(
                account_state=account_state,
                dry_run=dry_run,
            )

            # Build request
            from shorts_engine.adapters.publisher.base import PublishRequest

            request = PublishRequest(
                video_path=video_path,
                title=video_job.title or "Untitled Short",
                description=video_job.description,
                tags=video_job.plan_data.get("tags", []) if video_job.plan_data else None,
                visibility=visibility,
            )

            # Dry run
            if dry_run:
                payload = build_tiktok_dry_run_payload(
                    video_path=str(video_path),
                    title=request.title,
                    description=request.description,
                    tags=request.tags,
                )
                publish_job.dry_run_payload = payload
                publish_job.status = "dry_run_complete"
                session.commit()

                return {
                    "success": True,
                    "publish_job_id": str(publish_job.id),
                    "dry_run": True,
                    "payload": payload,
                }

            # Publish
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(publisher.publish(request))
            finally:
                loop.close()

            # Handle result
            if result.success:
                metadata = result.metadata or {}

                # Check for NEEDS_MANUAL_PUBLISH
                if metadata.get("needs_manual_publish"):
                    publish_job.status = PublishStatus.NEEDS_MANUAL_PUBLISH
                    publish_job.manual_publish_path = metadata.get("manual_publish_path")
                    publish_job.share_url = metadata.get("share_url")
                    publish_job.api_response = metadata
                    session.commit()

                    logger.warning(
                        f"TikTok Direct Post not available for account '{account_label}'. "
                        f"Video stored at: {publish_job.manual_publish_path}"
                    )

                    return {
                        "success": True,
                        "publish_job_id": str(publish_job.id),
                        "status": "needs_manual_publish",
                        "manual_publish_path": publish_job.manual_publish_path,
                        "share_url": publish_job.share_url,
                        "message": "Direct Post not available. Use TikTok app to upload.",
                    }

                # Normal success
                publish_job.status = "published"
                publish_job.platform_video_id = result.platform_video_id
                publish_job.platform_url = result.url
                publish_job.actual_publish_at = datetime.now(timezone.utc)
                publish_job.api_response = result.metadata

                increment_upload_count(session, account.id)
                session.commit()

                logger.info(
                    f"Published video to TikTok: {result.url} "
                    f"(video ID: {result.platform_video_id})"
                )

                return {
                    "success": True,
                    "publish_job_id": str(publish_job.id),
                    "platform_video_id": result.platform_video_id,
                    "url": result.url,
                }
            else:
                publish_job.status = "failed"
                publish_job.error_message = result.error_message
                publish_job.api_response = result.metadata
                session.commit()

                return {
                    "success": False,
                    "publish_job_id": str(publish_job.id),
                    "error": result.error_message,
                }

        except TikTokOAuthError as e:
            if "expired" in str(e).lower() or "invalid" in str(e).lower():
                mark_account_revoked(session, account.id, str(e))

            publish_job.status = "failed"
            publish_job.error_message = str(e)
            session.commit()

            return {
                "success": False,
                "publish_job_id": str(publish_job.id),
                "error": str(e),
            }

        except Exception as e:
            logger.exception(f"TikTok publishing failed: {e}")

            publish_job.status = "failed"
            publish_job.error_message = str(e)
            publish_job.retry_count = (publish_job.retry_count or 0) + 1
            session.commit()

            raise


@shared_task(bind=True)
def run_publish_pipeline_task(
    self,
    video_job_id: str,
    youtube_account: str | None = None,
    instagram_account: str | None = None,
    tiktok_account: str | None = None,
    destinations: list[dict[str, str]] | None = None,
    scheduled_publish_at: str | None = None,
    visibility: str = "public",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full publish pipeline for multiple platforms.

    Supports both legacy single-platform parameters and new multi-destination format.
    Respects the AUTOPUBLISH_ENABLED setting - if disabled, blocks automatic publishing.

    Args:
        video_job_id: UUID of the video job to publish.
        youtube_account: Label of YouTube account to use (legacy).
        instagram_account: Label of Instagram account to use (legacy).
        tiktok_account: Label of TikTok account to use (legacy).
        destinations: List of dicts with {platform, label} for multi-destination.
        scheduled_publish_at: Optional ISO 8601 datetime for scheduling.
        visibility: Video visibility.
        dry_run: If True, don't actually upload.

    Returns:
        Dict with publish results for each platform.
    """
    # Autopublish guard - block automatic publishing if disabled
    if not settings.autopublish_enabled and not dry_run:
        logger.warning(
            "autopublish_blocked",
            video_job_id=video_job_id,
            reason="AUTOPUBLISH_ENABLED=false",
        )
        return {
            "success": False,
            "skipped": True,
            "reason": "Autopublish disabled. Set AUTOPUBLISH_ENABLED=true to enable automatic publishing.",
            "video_job_id": video_job_id,
        }

    results = {}

    # Build destination list from both legacy and new format
    dest_list: list[dict[str, str]] = []

    if destinations:
        dest_list.extend(destinations)

    # Add legacy parameters
    if youtube_account:
        dest_list.append({"platform": "youtube", "label": youtube_account})
    if instagram_account:
        dest_list.append({"platform": "instagram", "label": instagram_account})
    if tiktok_account:
        dest_list.append({"platform": "tiktok", "label": tiktok_account})

    if not dest_list:
        return {
            "success": False,
            "error": "No destinations specified. Provide at least one platform account.",
        }

    # Dispatch to platform-specific tasks
    for dest in dest_list:
        platform = dest.get("platform", "").lower()
        label = dest.get("label", "")

        if not platform or not label:
            results[f"{platform}:{label}"] = {
                "success": False,
                "error": "Invalid destination format",
            }
            continue

        try:
            if platform == "youtube":
                result = publish_to_youtube_task.apply(
                    args=[video_job_id, label, scheduled_publish_at, visibility, dry_run]
                ).get()
            elif platform == "instagram":
                result = publish_to_instagram_task.apply(
                    args=[video_job_id, label, scheduled_publish_at, visibility, dry_run]
                ).get()
            elif platform == "tiktok":
                result = publish_to_tiktok_task.apply(
                    args=[video_job_id, label, scheduled_publish_at, visibility, dry_run]
                ).get()
            else:
                result = {"success": False, "error": f"Unknown platform: {platform}"}

            results[f"{platform}:{label}"] = result

        except Exception as e:
            logger.exception(f"Failed to publish to {platform}:{label}: {e}")
            results[f"{platform}:{label}"] = {
                "success": False,
                "error": str(e),
            }

    # Calculate overall success
    all_success = all(r.get("success") for r in results.values()) if results else False
    any_success = any(r.get("success") for r in results.values()) if results else False

    return {
        "success": all_success,
        "partial_success": any_success and not all_success,
        "video_job_id": video_job_id,
        "platforms": results,
    }


@shared_task(bind=True)
def check_publish_status_task(
    self,
    publish_job_id: str,
) -> dict[str, Any]:
    """Check the status of a published video on the platform.

    Args:
        publish_job_id: UUID of the publish job.

    Returns:
        Dict with current status.
    """
    with get_session_context() as session:
        publish_job = session.get(PublishJobModel, UUID(publish_job_id))
        if not publish_job:
            return {"success": False, "error": f"Publish job not found: {publish_job_id}"}

        if publish_job.platform != "youtube":
            return {"success": False, "error": f"Unsupported platform: {publish_job.platform}"}

        if not publish_job.platform_video_id:
            return {
                "success": False,
                "error": "No platform video ID - video may not have been uploaded yet",
            }

        try:
            account_state = get_account_state(session, publish_job.account_id)
            publisher = YouTubePublisher(account_state=account_state)

            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                status = loop.run_until_complete(
                    publisher.get_video_status(publish_job.platform_video_id)
                )
            finally:
                loop.close()

            return {
                "success": True,
                "publish_job_id": str(publish_job.id),
                "platform_video_id": publish_job.platform_video_id,
                "status": status,
            }

        except Exception as e:
            return {
                "success": False,
                "publish_job_id": str(publish_job.id),
                "error": str(e),
            }
