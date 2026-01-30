"""Project management endpoints."""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from shorts_engine.db.models import ProjectModel
from shorts_engine.db.session import get_session_context
from shorts_engine.logging import get_logger

router = APIRouter(prefix="/projects", tags=["Projects"])
logger = get_logger(__name__)


class CreateProjectRequest(BaseModel):
    """Request to create a project."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    default_style_preset: str = Field(default="educational", max_length=100)
    target_platforms: list[str] = Field(default=["youtube"])
    settings: dict = Field(default_factory=dict)


class UpdateProjectRequest(BaseModel):
    """Request to update a project."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    default_style_preset: str | None = Field(None, max_length=100)
    target_platforms: list[str] | None = None
    settings: dict | None = None
    is_active: bool | None = None


class ProjectResponse(BaseModel):
    """Project response model."""

    id: str
    name: str
    description: str | None
    default_style_preset: str | None
    target_platforms: list[str]
    settings: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


def _model_to_response(project: ProjectModel) -> ProjectResponse:
    """Convert a ProjectModel to ProjectResponse."""
    settings = project.settings or {}
    target_platforms = settings.get("target_platforms", ["youtube"])
    return ProjectResponse(
        id=str(project.id),
        name=project.name,
        description=project.description,
        default_style_preset=project.default_style_preset,
        target_platforms=target_platforms,
        settings=settings,
        is_active=project.is_active,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create project",
    description="Create a new project (content brand / channel).",
)
async def create_project(request: CreateProjectRequest) -> ProjectResponse:
    """Create a new project."""
    logger.info("create_project", name=request.name)

    with get_session_context() as session:
        # Store target_platforms in settings
        settings = request.settings.copy()
        settings["target_platforms"] = request.target_platforms

        project = ProjectModel(
            name=request.name,
            description=request.description,
            default_style_preset=request.default_style_preset,
            settings=settings,
        )
        session.add(project)
        session.commit()
        session.refresh(project)

        logger.info("project_created", project_id=str(project.id))
        return _model_to_response(project)


@router.get(
    "",
    response_model=list[ProjectResponse],
    summary="List projects",
    description="List all projects.",
)
async def list_projects(
    active_only: bool = True,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[ProjectResponse]:
    """List all projects."""
    with get_session_context() as session:
        query = select(ProjectModel).order_by(ProjectModel.created_at.desc())
        if active_only:
            query = query.where(ProjectModel.is_active == True)  # noqa: E712
        query = query.limit(limit).offset(offset)

        result = session.execute(query)
        projects = result.scalars().all()

        return [_model_to_response(p) for p in projects]


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get project",
    description="Get a project by ID.",
)
async def get_project(project_id: str) -> ProjectResponse:
    """Get a project by ID."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )

    with get_session_context() as session:
        project = session.get(ProjectModel, project_uuid)

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )

        return _model_to_response(project)


@router.put(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Update project",
    description="Update a project by ID.",
)
async def update_project(project_id: str, request: UpdateProjectRequest) -> ProjectResponse:
    """Update a project."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )

    with get_session_context() as session:
        project = session.get(ProjectModel, project_uuid)

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )

        # Update fields if provided
        if request.name is not None:
            project.name = request.name
        if request.description is not None:
            project.description = request.description
        if request.default_style_preset is not None:
            project.default_style_preset = request.default_style_preset
        if request.is_active is not None:
            project.is_active = request.is_active

        # Handle settings and target_platforms
        if request.settings is not None or request.target_platforms is not None:
            settings = (project.settings or {}).copy()
            if request.settings is not None:
                settings.update(request.settings)
            if request.target_platforms is not None:
                settings["target_platforms"] = request.target_platforms
            project.settings = settings

        session.commit()
        session.refresh(project)

        logger.info("project_updated", project_id=str(project.id))
        return _model_to_response(project)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project",
    description="Delete a project by ID. This will cascade delete all related video jobs.",
)
async def delete_project(project_id: str) -> None:
    """Delete a project."""
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )

    with get_session_context() as session:
        project = session.get(ProjectModel, project_uuid)

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )

        session.delete(project)
        session.commit()

        logger.info("project_deleted", project_id=project_id)
