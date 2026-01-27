"""Platform account management endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from shorts_engine.db.session import get_session_context
from shorts_engine.logging import get_logger
from shorts_engine.services.accounts import (
    AccountError,
    AccountNotFoundError,
    check_upload_limit,
    get_account_by_id,
    get_account_by_label,
    link_account_to_project,
    list_accounts,
)

router = APIRouter(prefix="/accounts", tags=["Accounts"])
logger = get_logger(__name__)


class AccountResponse(BaseModel):
    """Response with account details."""

    id: str
    platform: str
    label: str
    external_id: str | None = None
    external_name: str | None = None
    status: str
    uploads_today: int
    max_uploads_per_day: int
    created_at: str | None = None


class AccountListResponse(BaseModel):
    """Response with list of accounts."""

    accounts: list[AccountResponse]
    total: int


class LinkAccountRequest(BaseModel):
    """Request to link an account to a project."""

    account_id: str = Field(..., description="Account UUID")
    project_id: str = Field(..., description="Project UUID")
    is_default: bool = Field(default=False, description="Set as default for this project")


class UploadLimitResponse(BaseModel):
    """Response with upload limit info."""

    can_upload: bool
    uploads_today: int
    max_uploads: int
    account_label: str


@router.get(
    "",
    response_model=AccountListResponse,
    summary="List accounts",
    description="List all connected platform accounts.",
)
async def list_platform_accounts(
    platform: str | None = None,
    status_filter: str | None = None,
) -> AccountListResponse:
    """List connected platform accounts."""
    from shorts_engine.config import settings

    with get_session_context() as session:
        accounts = list_accounts(session, platform=platform, status=status_filter)

        return AccountListResponse(
            accounts=[
                AccountResponse(
                    id=str(account.id),
                    platform=account.platform,
                    label=account.label,
                    external_id=account.external_id,
                    external_name=account.external_name,
                    status=account.status,
                    uploads_today=account.uploads_today or 0,
                    max_uploads_per_day=settings.youtube_max_uploads_per_day,
                    created_at=account.created_at.isoformat() if account.created_at else None,
                )
                for account in accounts
            ],
            total=len(accounts),
        )


@router.get(
    "/{account_id}",
    response_model=AccountResponse,
    summary="Get account",
    description="Get details of a specific account.",
)
async def get_account(account_id: str) -> AccountResponse:
    """Get account details."""
    from shorts_engine.config import settings

    try:
        account_uuid = UUID(account_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid account ID",
        )

    with get_session_context() as session:
        try:
            account = get_account_by_id(session, account_uuid)
        except AccountNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )

        return AccountResponse(
            id=str(account.id),
            platform=account.platform,
            label=account.label,
            external_id=account.external_id,
            external_name=account.external_name,
            status=account.status,
            uploads_today=account.uploads_today or 0,
            max_uploads_per_day=settings.youtube_max_uploads_per_day,
            created_at=account.created_at.isoformat() if account.created_at else None,
        )


@router.get(
    "/label/{platform}/{label}",
    response_model=AccountResponse,
    summary="Get account by label",
    description="Get account by platform and label.",
)
async def get_account_by_platform_label(platform: str, label: str) -> AccountResponse:
    """Get account by platform and label."""
    from shorts_engine.config import settings

    with get_session_context() as session:
        try:
            account = get_account_by_label(session, platform, label)
        except AccountNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {platform} account found with label '{label}'",
            )

        return AccountResponse(
            id=str(account.id),
            platform=account.platform,
            label=account.label,
            external_id=account.external_id,
            external_name=account.external_name,
            status=account.status,
            uploads_today=account.uploads_today or 0,
            max_uploads_per_day=settings.youtube_max_uploads_per_day,
            created_at=account.created_at.isoformat() if account.created_at else None,
        )


@router.post(
    "/link",
    response_model=dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Link account to project",
    description="Link a platform account to a project for publishing.",
)
async def link_account(request: LinkAccountRequest) -> dict[str, Any]:
    """Link an account to a project."""
    try:
        account_uuid = UUID(request.account_id)
        project_uuid = UUID(request.project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid account ID or project ID",
        )

    with get_session_context() as session:
        try:
            link = link_account_to_project(
                session,
                account_uuid,
                project_uuid,
                is_default=request.is_default,
            )

            return {
                "success": True,
                "link_id": str(link.id),
                "account_id": str(link.account_id),
                "project_id": str(link.project_id),
                "is_default": link.is_default,
            }

        except AccountNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        except AccountError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )


@router.get(
    "/{account_id}/upload-limit",
    response_model=UploadLimitResponse,
    summary="Check upload limit",
    description="Check if account can upload (daily limit check).",
)
async def check_account_upload_limit(account_id: str) -> UploadLimitResponse:
    """Check upload limit for an account."""
    try:
        account_uuid = UUID(account_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid account ID",
        )

    with get_session_context() as session:
        try:
            account = get_account_by_id(session, account_uuid)
            can_upload, uploads_today, max_uploads = check_upload_limit(session, account_uuid)

            return UploadLimitResponse(
                can_upload=can_upload,
                uploads_today=uploads_today,
                max_uploads=max_uploads,
                account_label=account.label,
            )

        except AccountNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )
