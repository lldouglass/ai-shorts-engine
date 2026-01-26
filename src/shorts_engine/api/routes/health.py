"""Health check endpoints."""

from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

from shorts_engine.api.deps import PipelineServiceDep
from shorts_engine.config import settings
from shorts_engine.logging import get_logger

router = APIRouter(tags=["Health"])
logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: dict[str, bool] | None = None


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    database: bool
    redis: bool
    components: dict[str, bool] | None = None


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Basic health check endpoint that verifies the API is running.",
)
async def health_check() -> HealthResponse:
    """Basic health check - is the API up?"""
    from shorts_engine import __version__

    return HealthResponse(
        status="healthy",
        version=__version__,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Comprehensive readiness check that verifies all dependencies.",
)
async def readiness_check(pipeline: PipelineServiceDep) -> ReadinessResponse:
    """Comprehensive readiness check including dependencies."""
    # Check database
    database_ok = False
    try:
        from sqlalchemy import text

        from shorts_engine.db.session import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            database_ok = True
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))

    # Check Redis
    redis_ok = False
    try:
        import redis

        r = redis.from_url(settings.redis_url)
        r.ping()
        redis_ok = True
    except Exception as e:
        logger.error("redis_health_check_failed", error=str(e))

    # Check pipeline components
    components = await pipeline.health_check()

    ready = database_ok and redis_ok and all(components.values())

    return ReadinessResponse(
        ready=ready,
        database=database_ok,
        redis=redis_ok,
        components=components,
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Simple liveness probe for Kubernetes.",
)
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe - is the process alive?"""
    return {"status": "alive"}
