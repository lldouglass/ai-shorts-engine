"""Monitoring and dashboard endpoints for operational visibility."""

from typing import Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from shorts_engine.db.session import get_session
from shorts_engine.logging import get_logger
from shorts_engine.services.metrics import DashboardMetrics, estimate_total_cost

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])
logger = get_logger(__name__)


class QueueDepthsResponse(BaseModel):
    """Queue depths by pipeline stage."""

    depths: dict[str, int] = Field(description="Job count by stage")


class SuccessRateItem(BaseModel):
    """Success rate for a single stage."""

    success_rate: float = Field(ge=0.0, le=1.0, description="Success rate (0.0-1.0)")
    total: int = Field(ge=0, description="Total jobs in time window")


class SuccessRatesResponse(BaseModel):
    """Success rates by pipeline stage."""

    rates: dict[str, SuccessRateItem] = Field(description="Success rates by stage")
    time_window_hours: int = Field(description="Time window in hours")


class GenerationTimesResponse(BaseModel):
    """Average generation times by stage."""

    times: dict[str, float] = Field(description="Average duration in seconds by stage")
    time_window_hours: int = Field(description="Time window in hours")


class CostSummaryResponse(BaseModel):
    """Cost summary for the time window."""

    total_cost_usd: float = Field(ge=0.0, description="Total estimated cost in USD")
    job_count: int = Field(ge=0, description="Number of jobs in window")
    avg_cost_per_job_usd: float = Field(ge=0.0, description="Average cost per job")
    estimated_cost_per_video: float = Field(ge=0.0, description="Current estimate per video")
    time_window_hours: int = Field(description="Time window in hours")


class QAStatsResponse(BaseModel):
    """QA check statistics."""

    pass_rate: float = Field(ge=0.0, le=1.0, description="QA pass rate (0.0-1.0)")
    total_checks: int = Field(ge=0, description="Total QA checks")
    passed: int = Field(ge=0, description="Number of passed checks")
    failed: int = Field(ge=0, description="Number of failed checks")
    avg_hook_clarity: float = Field(ge=0.0, le=1.0, description="Average hook clarity score")
    avg_coherence: float = Field(ge=0.0, le=1.0, description="Average coherence score")
    time_window_hours: int = Field(description="Time window in hours")


class DashboardSummaryResponse(BaseModel):
    """Combined dashboard summary."""

    queue_depths: dict[str, int] = Field(description="Current queue depths by stage")
    success_rates: dict[str, SuccessRateItem] = Field(description="Success rates by stage")
    generation_times: dict[str, float] = Field(description="Avg generation times by stage")
    cost_summary: CostSummaryResponse = Field(description="Cost summary")
    qa_stats: QAStatsResponse = Field(description="QA statistics")
    time_window_hours: int = Field(description="Time window for rate calculations")


@router.get(
    "/queue-depths",
    response_model=QueueDepthsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get queue depths",
    description="Get current pending job counts by pipeline stage.",
)
async def get_queue_depths(
    session: Session = Depends(get_session),
) -> QueueDepthsResponse:
    """Get current queue depths by stage."""
    metrics = DashboardMetrics(session)
    depths = metrics.get_queue_depths()
    return QueueDepthsResponse(depths=depths)


@router.get(
    "/success-rates",
    response_model=SuccessRatesResponse,
    status_code=status.HTTP_200_OK,
    summary="Get success rates",
    description="Get job success/failure rates by stage over time window.",
)
async def get_success_rates(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    session: Session = Depends(get_session),
) -> SuccessRatesResponse:
    """Get success rates by stage."""
    metrics = DashboardMetrics(session)
    rates_data = metrics.get_success_rates(hours)

    # Convert to response model
    rates = {
        stage: SuccessRateItem(**data) for stage, data in rates_data.items()
    }

    return SuccessRatesResponse(rates=rates, time_window_hours=hours)


@router.get(
    "/generation-times",
    response_model=GenerationTimesResponse,
    status_code=status.HTTP_200_OK,
    summary="Get generation times",
    description="Get average job duration by stage over time window.",
)
async def get_generation_times(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    session: Session = Depends(get_session),
) -> GenerationTimesResponse:
    """Get average generation times by stage."""
    metrics = DashboardMetrics(session)
    times = metrics.get_generation_times(hours)
    return GenerationTimesResponse(times=times, time_window_hours=hours)


@router.get(
    "/cost-summary",
    response_model=CostSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get cost summary",
    description="Get estimated costs over time window.",
)
async def get_cost_summary(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    session: Session = Depends(get_session),
) -> CostSummaryResponse:
    """Get cost summary."""
    metrics = DashboardMetrics(session)
    summary = metrics.get_cost_summary(hours)
    return CostSummaryResponse(
        total_cost_usd=summary["total_cost_usd"],
        job_count=summary["job_count"],
        avg_cost_per_job_usd=summary["avg_cost_per_job_usd"],
        estimated_cost_per_video=summary["estimated_cost_per_video"],
        time_window_hours=hours,
    )


@router.get(
    "/qa-stats",
    response_model=QAStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get QA statistics",
    description="Get QA check pass rates and average scores.",
)
async def get_qa_stats(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    session: Session = Depends(get_session),
) -> QAStatsResponse:
    """Get QA statistics."""
    metrics = DashboardMetrics(session)
    stats = metrics.get_qa_stats(hours)
    return QAStatsResponse(
        pass_rate=stats["pass_rate"],
        total_checks=stats["total_checks"],
        passed=stats["passed"],
        failed=stats["failed"],
        avg_hook_clarity=stats["avg_hook_clarity"],
        avg_coherence=stats["avg_coherence"],
        time_window_hours=hours,
    )


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get dashboard summary",
    description="Get combined dashboard data in a single request.",
)
async def get_dashboard_summary(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    session: Session = Depends(get_session),
) -> DashboardSummaryResponse:
    """Get combined dashboard summary."""
    metrics = DashboardMetrics(session)
    summary = metrics.get_summary(hours)

    # Convert nested dicts to response models
    success_rates = {
        stage: SuccessRateItem(**data)
        for stage, data in summary["success_rates"].items()
    }

    cost_summary = CostSummaryResponse(
        total_cost_usd=summary["cost_summary"]["total_cost_usd"],
        job_count=summary["cost_summary"]["job_count"],
        avg_cost_per_job_usd=summary["cost_summary"]["avg_cost_per_job_usd"],
        estimated_cost_per_video=summary["cost_summary"]["estimated_cost_per_video"],
        time_window_hours=hours,
    )

    qa_stats = QAStatsResponse(
        pass_rate=summary["qa_stats"]["pass_rate"],
        total_checks=summary["qa_stats"]["total_checks"],
        passed=summary["qa_stats"]["passed"],
        failed=summary["qa_stats"]["failed"],
        avg_hook_clarity=summary["qa_stats"]["avg_hook_clarity"],
        avg_coherence=summary["qa_stats"]["avg_coherence"],
        time_window_hours=hours,
    )

    return DashboardSummaryResponse(
        queue_depths=summary["queue_depths"],
        success_rates=success_rates,
        generation_times=summary["generation_times"],
        cost_summary=cost_summary,
        qa_stats=qa_stats,
        time_window_hours=hours,
    )
