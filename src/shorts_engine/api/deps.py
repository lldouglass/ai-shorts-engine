"""FastAPI dependencies."""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from shorts_engine.db.session import get_session
from shorts_engine.services.pipeline import PipelineService

# Database session dependency
SessionDep = Annotated[Session, Depends(get_session)]


def get_pipeline_service() -> PipelineService:
    """Get the pipeline service instance."""
    return PipelineService()


PipelineServiceDep = Annotated[PipelineService, Depends(get_pipeline_service)]
