"""Metrics collection and dashboard queries for pipeline monitoring."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from shorts_engine.config import settings
from shorts_engine.db.models import PipelineMetricModel, VideoJobModel
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    labels: dict[str, str] | None = None
    video_job_id: UUID | None = None
    recorded_at: datetime | None = None


class MetricsCollector:
    """Collects and stores pipeline metrics.

    Provides Prometheus-style metrics collection with counter, gauge,
    and histogram support. Metrics are stored in PostgreSQL for querying.
    """

    def __init__(self, session: Session | None = None):
        """Initialize metrics collector.

        Args:
            session: Optional database session. If not provided,
                     creates new sessions for each operation.
        """
        self._session = session
        self._enabled = settings.metrics_enabled

    def _get_session(self) -> Session:
        """Get database session."""
        if self._session:
            return self._session
        raise RuntimeError("No session provided - use with get_session_context()")

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        video_job_id: UUID | None = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name (e.g., "pipeline_jobs_started_total")
            value: Amount to increment (default 1.0)
            labels: Optional labels (e.g., {"stage": "planning"})
            video_job_id: Optional associated video job
        """
        if not self._enabled:
            return
        self._record("counter", name, value, labels, video_job_id)

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        video_job_id: UUID | None = None,
    ) -> None:
        """Set a gauge metric value.

        Args:
            name: Metric name (e.g., "qa_hook_clarity")
            value: Current value
            labels: Optional labels
            video_job_id: Optional associated video job
        """
        if not self._enabled:
            return
        self._record("gauge", name, value, labels, video_job_id)

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        video_job_id: UUID | None = None,
    ) -> None:
        """Record a histogram observation.

        Args:
            name: Metric name (e.g., "pipeline_job_duration_seconds")
            value: Observed value
            labels: Optional labels
            video_job_id: Optional associated video job
        """
        if not self._enabled:
            return
        self._record("histogram", name, value, labels, video_job_id)

    def _record(
        self,
        metric_type: str,
        name: str,
        value: float,
        labels: dict[str, str] | None,
        video_job_id: UUID | None,
    ) -> None:
        """Record a metric to the database."""
        session = self._get_session()
        metric = PipelineMetricModel(
            id=uuid4(),
            metric_name=name,
            metric_type=metric_type,
            value=value,
            labels=labels,
            video_job_id=video_job_id,
            recorded_at=datetime.now(UTC),
        )
        session.add(metric)

    @contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        video_job_id: UUID | None = None,
    ) -> Generator[None, None, None]:
        """Context manager for timing operations.

        Usage:
            with metrics.timer("pipeline_job_duration_seconds", {"stage": "planning"}):
                do_work()

        Args:
            name: Metric name for duration
            labels: Optional labels
            video_job_id: Optional associated video job
        """
        start = time.time()
        try:
            yield
        finally:
            if self._enabled:
                duration = time.time() - start
                self.histogram(name, duration, labels, video_job_id)


# Convenience functions for recording metrics without explicit session
def record_job_started(
    session: Session,
    video_job_id: UUID,
    stage: str,
) -> None:
    """Record that a job has started a stage."""
    collector = MetricsCollector(session)
    collector.counter(
        "pipeline_jobs_started_total",
        labels={"stage": stage},
        video_job_id=video_job_id,
    )


def record_job_completed(
    session: Session,
    video_job_id: UUID,
    stage: str,
    duration_seconds: float | None = None,
) -> None:
    """Record that a job has completed a stage."""
    collector = MetricsCollector(session)
    collector.counter(
        "pipeline_jobs_completed_total",
        labels={"stage": stage},
        video_job_id=video_job_id,
    )
    if duration_seconds is not None:
        collector.histogram(
            "pipeline_job_duration_seconds",
            duration_seconds,
            labels={"stage": stage},
            video_job_id=video_job_id,
        )


def record_job_failed(
    session: Session,
    video_job_id: UUID,
    stage: str,
    error_type: str,
) -> None:
    """Record that a job has failed a stage."""
    collector = MetricsCollector(session)
    collector.counter(
        "pipeline_jobs_failed_total",
        labels={"stage": stage, "error_type": error_type},
        video_job_id=video_job_id,
    )


def record_qa_check(
    session: Session,
    video_job_id: UUID,
    stage: str,
    passed: bool,
    hook_clarity: float | None = None,
    coherence: float | None = None,
) -> None:
    """Record QA check results."""
    collector = MetricsCollector(session)
    collector.counter(
        "qa_checks_total",
        labels={"stage": stage, "status": "passed" if passed else "failed"},
        video_job_id=video_job_id,
    )
    if hook_clarity is not None:
        collector.gauge(
            "qa_hook_clarity",
            hook_clarity,
            labels={"stage": stage},
            video_job_id=video_job_id,
        )
    if coherence is not None:
        collector.gauge(
            "qa_coherence",
            coherence,
            labels={"stage": stage},
            video_job_id=video_job_id,
        )


def record_cost_estimate(
    session: Session,
    video_job_id: UUID,
    cost_usd: float,
) -> None:
    """Record estimated cost for a job."""
    collector = MetricsCollector(session)
    collector.gauge(
        "job_cost_estimate_usd",
        cost_usd,
        video_job_id=video_job_id,
    )


# Cost estimation functions
def estimate_planning_cost() -> float:
    """Estimate cost of planning stage (LLM calls)."""
    return 0.15  # ~$0.15 for planning LLM calls


def estimate_generation_cost(scene_count: int = 7) -> float:
    """Estimate cost of video generation (Luma AI)."""
    return scene_count * 0.50  # ~$0.50 per scene at Luma


def estimate_rendering_cost() -> float:
    """Estimate cost of rendering (Creatomate + ElevenLabs)."""
    return 0.40  # ~$0.25 Creatomate + ~$0.15 ElevenLabs


def estimate_total_cost(scene_count: int = 7) -> float:
    """Estimate total cost per video."""
    return (
        estimate_planning_cost() + estimate_generation_cost(scene_count) + estimate_rendering_cost()
    )


class DashboardMetrics:
    """Aggregated queries for monitoring dashboard."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def get_queue_depths(self) -> dict[str, int]:
        """Get current queue depths by stage.

        Returns:
            Dict mapping stage names to pending job counts
        """
        result = (
            self.session.query(
                VideoJobModel.stage,
                func.count(VideoJobModel.id).label("count"),
            )
            .filter(VideoJobModel.status == "pending")
            .group_by(VideoJobModel.stage)
            .all()
        )
        return {row.stage: row.count for row in result}  # type: ignore[misc]

    def get_success_rates(
        self,
        hours: int = 24,
    ) -> dict[str, dict[str, float]]:
        """Get success rates by stage over time window.

        Args:
            hours: Time window in hours (default 24)

        Returns:
            Dict mapping stage to {"success_rate": float, "total": int}
        """
        since = datetime.now(UTC) - timedelta(hours=hours)

        # Get completed counts by stage
        completed = (
            self.session.query(
                PipelineMetricModel.labels["stage"].astext.label("stage"),
                func.count().label("count"),
            )
            .filter(
                PipelineMetricModel.metric_name == "pipeline_jobs_completed_total",
                PipelineMetricModel.recorded_at >= since,
            )
            .group_by(PipelineMetricModel.labels["stage"].astext)
            .all()
        )
        completed_map = {row.stage: row.count for row in completed}

        # Get failed counts by stage
        failed = (
            self.session.query(
                PipelineMetricModel.labels["stage"].astext.label("stage"),
                func.count().label("count"),
            )
            .filter(
                PipelineMetricModel.metric_name == "pipeline_jobs_failed_total",
                PipelineMetricModel.recorded_at >= since,
            )
            .group_by(PipelineMetricModel.labels["stage"].astext)
            .all()
        )
        failed_map = {row.stage: row.count for row in failed}

        # Calculate rates
        all_stages = set(completed_map.keys()) | set(failed_map.keys())
        rates = {}
        for stage in all_stages:
            success = completed_map.get(stage, 0)
            fail = failed_map.get(stage, 0)
            total = success + fail
            rates[stage] = {
                "success_rate": success / total if total > 0 else 0.0,
                "total": total,
            }

        return rates

    def get_generation_times(
        self,
        hours: int = 24,
    ) -> dict[str, float]:
        """Get average generation times by stage.

        Args:
            hours: Time window in hours

        Returns:
            Dict mapping stage to average duration in seconds
        """
        since = datetime.now(UTC) - timedelta(hours=hours)

        result = (
            self.session.query(
                PipelineMetricModel.labels["stage"].astext.label("stage"),
                func.avg(PipelineMetricModel.value).label("avg_duration"),
            )
            .filter(
                PipelineMetricModel.metric_name == "pipeline_job_duration_seconds",
                PipelineMetricModel.recorded_at >= since,
            )
            .group_by(PipelineMetricModel.labels["stage"].astext)
            .all()
        )

        return {row.stage: float(row.avg_duration) for row in result}

    def get_cost_summary(
        self,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get cost summary over time window.

        Args:
            hours: Time window in hours

        Returns:
            Dict with total_cost, job_count, avg_cost_per_job
        """
        since = datetime.now(UTC) - timedelta(hours=hours)

        result = (
            self.session.query(
                func.sum(PipelineMetricModel.value).label("total"),
                func.count(func.distinct(PipelineMetricModel.video_job_id)).label("jobs"),
            )
            .filter(
                PipelineMetricModel.metric_name == "job_cost_estimate_usd",
                PipelineMetricModel.recorded_at >= since,
            )
            .first()
        )

        total = float(result.total) if result and result.total else 0.0
        jobs = result.jobs if result else 0
        avg = total / jobs if jobs > 0 else 0.0

        return {
            "total_cost_usd": total,
            "job_count": jobs,
            "avg_cost_per_job_usd": avg,
            "estimated_cost_per_video": estimate_total_cost(),
        }

    def get_qa_stats(
        self,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get QA check statistics.

        Args:
            hours: Time window in hours

        Returns:
            Dict with pass_rate, total_checks, avg_hook_clarity, avg_coherence
        """
        since = datetime.now(UTC) - timedelta(hours=hours)

        # Get pass/fail counts
        checks = (
            self.session.query(
                PipelineMetricModel.labels["status"].astext.label("status"),
                func.count().label("count"),
            )
            .filter(
                PipelineMetricModel.metric_name == "qa_checks_total",
                PipelineMetricModel.recorded_at >= since,
            )
            .group_by(PipelineMetricModel.labels["status"].astext)
            .all()
        )
        passed = sum(row.count for row in checks if row.status == "passed")
        failed = sum(row.count for row in checks if row.status == "failed")
        total = passed + failed

        # Get average scores
        hook_clarity = (
            self.session.query(func.avg(PipelineMetricModel.value))
            .filter(
                PipelineMetricModel.metric_name == "qa_hook_clarity",
                PipelineMetricModel.recorded_at >= since,
            )
            .scalar()
        )
        coherence = (
            self.session.query(func.avg(PipelineMetricModel.value))
            .filter(
                PipelineMetricModel.metric_name == "qa_coherence",
                PipelineMetricModel.recorded_at >= since,
            )
            .scalar()
        )

        return {
            "pass_rate": passed / total if total > 0 else 0.0,
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "avg_hook_clarity": float(hook_clarity) if hook_clarity else 0.0,
            "avg_coherence": float(coherence) if coherence else 0.0,
        }

    def get_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get combined dashboard summary.

        Args:
            hours: Time window in hours

        Returns:
            Combined dict with all dashboard metrics
        """
        return {
            "queue_depths": self.get_queue_depths(),
            "success_rates": self.get_success_rates(hours),
            "generation_times": self.get_generation_times(hours),
            "cost_summary": self.get_cost_summary(hours),
            "qa_stats": self.get_qa_stats(hours),
            "time_window_hours": hours,
        }


def cleanup_old_metrics(session: Session, days: int | None = None) -> int:
    """Delete metrics older than retention period.

    Args:
        session: Database session
        days: Days to retain (default from settings)

    Returns:
        Number of deleted records
    """
    retention_days = days or settings.metrics_retention_days
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)

    result = (
        session.query(PipelineMetricModel)
        .filter(PipelineMetricModel.recorded_at < cutoff)
        .delete(synchronize_session=False)
    )

    logger.info(
        "metrics_cleanup_completed",
        deleted_count=result,
        retention_days=retention_days,
    )

    return result
