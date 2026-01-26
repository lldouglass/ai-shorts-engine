"""Platform account management service.

Handles connecting, storing, and managing platform accounts (YouTube, TikTok, etc.).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from shorts_engine.adapters.publisher.youtube import YouTubeAccountState
from shorts_engine.adapters.publisher.youtube_oauth import (
    OAuthCredentials,
    OAuthError,
    refresh_access_token,
    run_device_flow,
    run_local_callback_flow,
)
from shorts_engine.config import settings
from shorts_engine.db.models import (
    AccountProjectModel,
    PlatformAccountModel,
    ProjectModel,
)
from shorts_engine.services.encryption import decrypt_token, encrypt_token, EncryptionError

logger = logging.getLogger(__name__)


class AccountNotFoundError(Exception):
    """Raised when an account is not found."""

    pass


class AccountError(Exception):
    """Raised for general account errors."""

    pass


def connect_youtube_account(
    session: Session,
    label: str,
    use_device_flow: bool = True,
) -> PlatformAccountModel:
    """Connect a YouTube account using OAuth.

    Args:
        session: Database session.
        label: User-friendly label for the account (e.g., "Main Channel").
        use_device_flow: If True, use device flow. Otherwise, use local callback.

    Returns:
        The created PlatformAccountModel.

    Raises:
        OAuthError: If OAuth flow fails.
        AccountError: If account already exists with same label.
    """
    # Check if account with this label already exists
    existing = session.execute(
        select(PlatformAccountModel).where(
            PlatformAccountModel.platform == "youtube",
            PlatformAccountModel.label == label,
        )
    ).scalar_one_or_none()

    if existing:
        raise AccountError(
            f"YouTube account with label '{label}' already exists. "
            "Use a different label or disconnect the existing account first."
        )

    # Run OAuth flow
    if use_device_flow:
        credentials = run_device_flow()
    else:
        credentials = run_local_callback_flow()

    # Create account record
    account = PlatformAccountModel(
        id=uuid4(),
        platform="youtube",
        label=label,
        external_id=credentials.channel_id,
        external_name=credentials.channel_title,
        encrypted_refresh_token=encrypt_token(credentials.refresh_token),
        encrypted_access_token=encrypt_token(credentials.access_token),
        token_expires_at=datetime.now(timezone.utc) + timedelta(seconds=credentials.expires_in),
        scopes=credentials.scope,
        status="active",
        uploads_today=0,
        metadata_={
            "connected_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    session.add(account)
    session.commit()
    session.refresh(account)

    logger.info(
        f"Connected YouTube account: {credentials.channel_title} "
        f"(channel ID: {credentials.channel_id}) as '{label}'"
    )

    return account


def get_account_by_label(
    session: Session,
    platform: str,
    label: str,
) -> PlatformAccountModel:
    """Get a platform account by label.

    Args:
        session: Database session.
        platform: Platform name (e.g., "youtube").
        label: Account label.

    Returns:
        The PlatformAccountModel.

    Raises:
        AccountNotFoundError: If account not found.
    """
    account = session.execute(
        select(PlatformAccountModel).where(
            PlatformAccountModel.platform == platform,
            PlatformAccountModel.label == label,
        )
    ).scalar_one_or_none()

    if not account:
        raise AccountNotFoundError(f"No {platform} account found with label '{label}'")

    return account


def get_account_by_id(
    session: Session,
    account_id: UUID,
) -> PlatformAccountModel:
    """Get a platform account by ID.

    Args:
        session: Database session.
        account_id: Account UUID.

    Returns:
        The PlatformAccountModel.

    Raises:
        AccountNotFoundError: If account not found.
    """
    account = session.get(PlatformAccountModel, account_id)

    if not account:
        raise AccountNotFoundError(f"No account found with ID '{account_id}'")

    return account


def list_accounts(
    session: Session,
    platform: str | None = None,
    status: str | None = None,
) -> list[PlatformAccountModel]:
    """List platform accounts.

    Args:
        session: Database session.
        platform: Filter by platform (optional).
        status: Filter by status (optional).

    Returns:
        List of matching accounts.
    """
    query = select(PlatformAccountModel).order_by(PlatformAccountModel.created_at.desc())

    if platform:
        query = query.where(PlatformAccountModel.platform == platform)

    if status:
        query = query.where(PlatformAccountModel.status == status)

    return list(session.execute(query).scalars().all())


def disconnect_account(
    session: Session,
    account_id: UUID,
) -> None:
    """Disconnect (delete) a platform account.

    Args:
        session: Database session.
        account_id: Account UUID.

    Raises:
        AccountNotFoundError: If account not found.
    """
    account = get_account_by_id(session, account_id)
    session.delete(account)
    session.commit()

    logger.info(f"Disconnected {account.platform} account: {account.label}")


def link_account_to_project(
    session: Session,
    account_id: UUID,
    project_id: UUID,
    is_default: bool = False,
) -> AccountProjectModel:
    """Link an account to a project.

    Args:
        session: Database session.
        account_id: Account UUID.
        project_id: Project UUID.
        is_default: Whether this is the default account for the project.

    Returns:
        The AccountProjectModel.
    """
    # Verify account and project exist
    account = get_account_by_id(session, account_id)
    project = session.get(ProjectModel, project_id)

    if not project:
        raise AccountError(f"Project not found: {project_id}")

    # Check if link already exists
    existing = session.execute(
        select(AccountProjectModel).where(
            AccountProjectModel.account_id == account_id,
            AccountProjectModel.project_id == project_id,
        )
    ).scalar_one_or_none()

    if existing:
        # Update default status if needed
        if is_default and not existing.is_default:
            # Clear other defaults first
            _clear_default_accounts(session, project_id, account.platform)
            existing.is_default = True
            session.commit()
        return existing

    # If setting as default, clear other defaults first
    if is_default:
        _clear_default_accounts(session, project_id, account.platform)

    # Create link
    link = AccountProjectModel(
        id=uuid4(),
        account_id=account_id,
        project_id=project_id,
        is_default=is_default,
    )

    session.add(link)
    session.commit()
    session.refresh(link)

    return link


def _clear_default_accounts(
    session: Session,
    project_id: UUID,
    platform: str,
) -> None:
    """Clear default status for all accounts of a platform for a project."""
    # Get all links for this project
    links = session.execute(
        select(AccountProjectModel)
        .join(PlatformAccountModel)
        .where(
            AccountProjectModel.project_id == project_id,
            PlatformAccountModel.platform == platform,
            AccountProjectModel.is_default == True,
        )
    ).scalars().all()

    for link in links:
        link.is_default = False


def get_account_state(
    session: Session,
    account_id: UUID,
) -> YouTubeAccountState:
    """Get account state for YouTube publishing.

    Decrypts tokens and returns a state object usable by the publisher.

    Args:
        session: Database session.
        account_id: Account UUID.

    Returns:
        YouTubeAccountState ready for publishing.

    Raises:
        AccountNotFoundError: If account not found.
        AccountError: If account is not active or tokens invalid.
    """
    account = get_account_by_id(session, account_id)

    if account.status != "active":
        raise AccountError(
            f"Account '{account.label}' is not active (status: {account.status}). "
            "Please reconnect the account."
        )

    if not account.encrypted_refresh_token:
        raise AccountError(
            f"Account '{account.label}' has no refresh token. "
            "Please reconnect the account."
        )

    try:
        refresh_token = decrypt_token(account.encrypted_refresh_token)
        access_token = (
            decrypt_token(account.encrypted_access_token)
            if account.encrypted_access_token
            else ""
        )
    except EncryptionError as e:
        raise AccountError(f"Failed to decrypt tokens: {e}")

    return YouTubeAccountState(
        account_id=account.id,
        access_token=access_token,
        refresh_token=refresh_token,
        token_expires_at=account.token_expires_at,
        uploads_today=account.uploads_today or 0,
        uploads_reset_at=account.uploads_reset_at,
        max_uploads_per_day=settings.youtube_max_uploads_per_day,
    )


def update_account_tokens(
    session: Session,
    account_id: UUID,
    access_token: str,
    expires_at: datetime,
) -> None:
    """Update account tokens after refresh.

    Args:
        session: Database session.
        account_id: Account UUID.
        access_token: New access token.
        expires_at: Token expiration time.
    """
    account = get_account_by_id(session, account_id)
    account.encrypted_access_token = encrypt_token(access_token)
    account.token_expires_at = expires_at
    session.commit()


def increment_upload_count(
    session: Session,
    account_id: UUID,
) -> None:
    """Increment the upload count for an account.

    Also resets the counter if it's a new day.

    Args:
        session: Database session.
        account_id: Account UUID.
    """
    account = get_account_by_id(session, account_id)
    now = datetime.now(timezone.utc)

    # Reset counter if new day
    if account.uploads_reset_at is None or account.uploads_reset_at < now:
        account.uploads_today = 0
        # Reset at midnight UTC
        tomorrow = now.date() + timedelta(days=1)
        account.uploads_reset_at = datetime(
            tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc
        )

    account.uploads_today = (account.uploads_today or 0) + 1
    session.commit()


def check_upload_limit(
    session: Session,
    account_id: UUID,
) -> tuple[bool, int, int]:
    """Check if account has reached upload limit.

    Args:
        session: Database session.
        account_id: Account UUID.

    Returns:
        Tuple of (can_upload, uploads_today, max_uploads).
    """
    account = get_account_by_id(session, account_id)
    now = datetime.now(timezone.utc)

    uploads_today = account.uploads_today or 0
    max_uploads = settings.youtube_max_uploads_per_day

    # Reset counter if new day
    if account.uploads_reset_at is None or account.uploads_reset_at < now:
        uploads_today = 0

    can_upload = uploads_today < max_uploads

    return can_upload, uploads_today, max_uploads


def mark_account_revoked(
    session: Session,
    account_id: UUID,
    reason: str | None = None,
) -> None:
    """Mark an account as revoked (token invalid).

    Args:
        session: Database session.
        account_id: Account UUID.
        reason: Optional reason for revocation.
    """
    account = get_account_by_id(session, account_id)
    account.status = "revoked"
    account.metadata_ = account.metadata_ or {}
    account.metadata_["revoked_at"] = datetime.now(timezone.utc).isoformat()
    if reason:
        account.metadata_["revoke_reason"] = reason
    session.commit()

    logger.warning(f"Marked account '{account.label}' as revoked: {reason}")
