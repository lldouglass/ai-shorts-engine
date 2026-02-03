"""Autonomous pipeline orchestrator for the learning loop.

This module contains tasks that orchestrate the full autonomous video
generation loop: topic generation → batch planning → video generation →
rendering → publishing.
"""

from typing import Any
from uuid import UUID

from celery import chain, group

from shorts_engine.config import get_settings
from shorts_engine.db.models import PlannedBatchModel, ProjectModel, VideoJobModel
from shorts_engine.db.session import get_session_context
from shorts_engine.jobs.learning_jobs import plan_next_batch_task
from shorts_engine.jobs.publish_pipeline import run_publish_pipeline_task
from shorts_engine.jobs.render_pipeline import run_render_pipeline_task
from shorts_engine.jobs.video_pipeline import (
    generate_all_scenes_from_plan,
    mark_ready_for_render_task,
    plan_job_task,
    verify_assets_task,
)
from shorts_engine.logging import get_logger
from shorts_engine.services.learning.context import OptimizationContextBuilder
from shorts_engine.services.topic_generator import TopicGenerator
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def _get_llm_provider() -> Any:
    """Get the configured LLM provider."""
    from shorts_engine.config import get_settings

    settings = get_settings()

    if settings.llm_provider == "openai":
        from shorts_engine.adapters.llm import OpenAIProvider

        return OpenAIProvider(model=settings.openai_model)
    elif settings.llm_provider == "anthropic":
        from shorts_engine.adapters.llm import AnthropicProvider

        return AnthropicProvider()
    else:
        from shorts_engine.adapters.llm import StubLLMProvider

        return StubLLMProvider()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context.

    Used for calling async adapters from Celery tasks.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@celery_app.task(
    bind=True,
    name="autonomous.generate_topics",
    max_retries=3,
    default_retry_delay=60,
)
def generate_topics_task(
    self: Any,
    project_id: str,
    n: int = 10,
    temperature: float = 0.8,
) -> dict[str, Any]:
    """Generate topic ideas for a project.

    Args:
        project_id: Project UUID string
        n: Number of topics to generate
        temperature: Creativity level (0-1)

    Returns:
        Dict with generated topics
    """
    task_id = self.request.id
    logger.info(
        "generate_topics_started",
        task_id=task_id,
        project_id=project_id,
        n=n,
    )

    try:
        project_uuid = UUID(project_id)
    except ValueError as e:
        return {"success": False, "error": f"Invalid project ID: {e}"}

    settings = get_settings()

    with get_session_context() as session:
        # Verify project exists
        project = session.get(ProjectModel, project_uuid)
        if not project:
            return {"success": False, "error": f"Project not found: {project_id}"}

        # Get LLM provider if needed
        llm_provider = None
        if settings.topic_provider == "llm":
            llm_provider = _get_llm_provider()

        # Generate topics
        generator = TopicGenerator(session, llm_provider=llm_provider)
        topics = _run_async(generator.generate(project_uuid, n=n, temperature=temperature))

        topic_strings = [t.topic for t in topics]

        logger.info(
            "generate_topics_completed",
            task_id=task_id,
            project_id=project_id,
            topics_generated=len(topic_strings),
        )

        return {
            "success": True,
            "project_id": project_id,
            "topics": topic_strings,
            "count": len(topic_strings),
        }


@celery_app.task(
    bind=True,
    name="autonomous.run_pipeline_for_job",
    max_retries=2,
    default_retry_delay=30,
)
def run_pipeline_for_job_task(
    self: Any,
    video_job_id: str,
    auto_render: bool | None = None,
    auto_publish: bool | None = None,
) -> dict[str, Any]:
    """Run the video generation pipeline for an existing job.

    Unlike run_full_pipeline_task, this works with a job that was already
    created by plan_next_batch_task.

    Args:
        video_job_id: UUID of the existing video job
        auto_render: Override config for auto-chaining render (default: from config)
        auto_publish: Override config for auto-chaining publish (default: from config)

    Returns:
        Dict with pipeline result
    """
    task_id = self.request.id
    settings = get_settings()

    # Use config defaults if not overridden
    if auto_render is None:
        auto_render = settings.auto_chain_render
    if auto_publish is None:
        auto_publish = settings.auto_chain_publish

    logger.info(
        "run_pipeline_for_job_started",
        task_id=task_id,
        video_job_id=video_job_id,
        auto_render=auto_render,
        auto_publish=auto_publish,
    )

    try:
        job_uuid = UUID(video_job_id)
    except ValueError as e:
        return {"success": False, "error": f"Invalid job ID: {e}"}

    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            return {"success": False, "error": f"Job not found: {video_job_id}"}

        # Check if already completed or in progress
        if job.stage in ("ready_for_render", "rendering", "ready_to_publish", "published"):
            logger.info(
                "run_pipeline_for_job_skipped",
                video_job_id=video_job_id,
                stage=job.stage,
                reason="already_past_generation",
            )
            return {
                "success": True,
                "video_job_id": video_job_id,
                "skipped": True,
                "stage": job.stage,
            }

    # Build pipeline chain - start with video generation
    pipeline_tasks = [
        plan_job_task.s(video_job_id),
        generate_all_scenes_from_plan.s(video_job_id),
        verify_assets_task.s(),
        mark_ready_for_render_task.s(),
    ]

    pipeline = chain(*pipeline_tasks)

    # Start the video generation pipeline
    result = pipeline.apply_async()

    # If auto_render is enabled, chain render after generation completes
    if auto_render:
        # Schedule render to run after video generation
        render_task = chain(
            run_render_pipeline_task.s(video_job_id),
        )

        # If auto_publish is also enabled, chain publish after render
        if auto_publish and settings.autopublish_enabled:
            render_task = chain(
                run_render_pipeline_task.s(video_job_id),
                # Publish task needs to extract job_id from render result
                _chain_publish_after_render.s(),
            )

        # Link render to run after generation completes
        result.link(render_task)

    logger.info(
        "run_pipeline_for_job_dispatched",
        task_id=task_id,
        video_job_id=video_job_id,
        pipeline_task_id=result.id,
        auto_render=auto_render,
        auto_publish=auto_publish,
    )

    return {
        "success": True,
        "video_job_id": video_job_id,
        "pipeline_task_id": result.id,
        "auto_render_enabled": auto_render,
        "auto_publish_enabled": auto_publish and settings.autopublish_enabled,
    }


@celery_app.task(
    bind=True,
    name="autonomous.chain_publish_after_render",
)
def _chain_publish_after_render(
    _self: Any,
    render_result: dict[str, Any],
) -> dict[str, Any]:
    """Chain publish task after render completes.

    This is a bridge task that extracts the job ID from render result
    and triggers the publish pipeline.
    """
    if not render_result.get("success"):
        logger.warning(
            "chain_publish_skipped",
            reason="render_failed",
            error=render_result.get("error"),
        )
        return {"success": False, "error": "Render failed, skipping publish"}

    video_job_id = render_result.get("video_job_id")
    if not video_job_id:
        return {"success": False, "error": "No video_job_id in render result"}

    settings = get_settings()
    if not settings.autopublish_enabled:
        logger.info(
            "chain_publish_skipped",
            reason="autopublish_disabled",
            video_job_id=video_job_id,
        )
        return {
            "success": True,
            "video_job_id": video_job_id,
            "skipped": True,
            "reason": "autopublish_disabled",
        }

    # Trigger publish pipeline
    result = run_publish_pipeline_task.delay(video_job_id)

    return {
        "success": True,
        "video_job_id": video_job_id,
        "publish_task_id": result.id,
    }


@celery_app.task(
    bind=True,
    name="autonomous.run_batch_pipelines",
    max_retries=2,
)
def run_batch_pipelines_task(
    self: Any,
    batch_result: dict[str, Any],
) -> dict[str, Any]:
    """Run pipelines for all jobs in a batch.

    Dispatches parallel pipeline tasks for each job created in the batch.

    Args:
        batch_result: Result from plan_next_batch_task

    Returns:
        Dict with dispatch results
    """
    task_id = self.request.id

    if not batch_result.get("success"):
        logger.error(
            "run_batch_pipelines_skipped",
            task_id=task_id,
            reason="batch_planning_failed",
            error=batch_result.get("error"),
        )
        return {
            "success": False,
            "error": f"Batch planning failed: {batch_result.get('error')}",
        }

    job_ids = batch_result.get("job_ids", [])
    if not job_ids:
        return {
            "success": True,
            "jobs_dispatched": 0,
            "message": "No jobs in batch",
        }

    logger.info(
        "run_batch_pipelines_started",
        task_id=task_id,
        batch_id=batch_result.get("batch_id"),
        job_count=len(job_ids),
    )

    # Dispatch pipelines in parallel using a group
    pipeline_group = group(run_pipeline_for_job_task.s(job_id) for job_id in job_ids)
    group_result = pipeline_group.apply_async()

    logger.info(
        "run_batch_pipelines_dispatched",
        task_id=task_id,
        batch_id=batch_result.get("batch_id"),
        jobs_dispatched=len(job_ids),
        group_task_id=group_result.id,
    )

    return {
        "success": True,
        "batch_id": batch_result.get("batch_id"),
        "jobs_dispatched": len(job_ids),
        "job_ids": job_ids,
        "group_task_id": group_result.id,
    }


@celery_app.task(
    bind=True,
    name="autonomous.run_nightly_batch",
    max_retries=2,
    default_retry_delay=300,
)
def run_nightly_batch_task(
    self: Any,
    project_id: str,
    n: int | None = None,
    auto_render: bool | None = None,
    auto_publish: bool | None = None,
) -> dict[str, Any]:
    """Run the full autonomous nightly batch.

    This is the main entry point for autonomous operation. It:
    1. Generates topic ideas using the TopicGenerator
    2. Plans a batch with recipe sampling (exploit/explore)
    3. Dispatches video generation pipelines for all jobs
    4. Optionally chains render and publish pipelines

    Args:
        project_id: Project UUID string
        n: Number of videos to create (default from config)
        auto_render: Override config for auto-chaining render
        auto_publish: Override config for auto-chaining publish

    Returns:
        Dict with batch planning and dispatch results
    """
    task_id = self.request.id
    settings = get_settings()

    # Use config defaults if not overridden
    if n is None:
        n = settings.autonomous_batch_size
    if auto_render is None:
        auto_render = settings.auto_chain_render
    if auto_publish is None:
        auto_publish = settings.auto_chain_publish

    logger.info(
        "nightly_batch_started",
        task_id=task_id,
        project_id=project_id,
        n=n,
        auto_render=auto_render,
        auto_publish=auto_publish,
    )

    try:
        project_uuid = UUID(project_id)
    except ValueError as e:
        return {"success": False, "error": f"Invalid project ID: {e}"}

    # Check if autonomous mode is enabled
    if not settings.autonomous_enabled:
        logger.warning(
            "nightly_batch_skipped",
            task_id=task_id,
            project_id=project_id,
            reason="autonomous_disabled",
        )
        return {
            "success": False,
            "error": "Autonomous mode is disabled. Set AUTONOMOUS_ENABLED=true to enable.",
        }

    # Step 1: Generate topics
    with get_session_context() as session:
        project = session.get(ProjectModel, project_uuid)
        if not project:
            return {"success": False, "error": f"Project not found: {project_id}"}

        # Get LLM provider if needed
        llm_provider = None
        if settings.topic_provider == "llm":
            llm_provider = _get_llm_provider()

        generator = TopicGenerator(session, llm_provider=llm_provider)

        # Generate more topics than needed for fallback
        topics_result = _run_async(generator.generate(project_uuid, n=n * 2, temperature=0.8))
        topics = [t.topic for t in topics_result]

    if not topics:
        logger.error(
            "nightly_batch_no_topics",
            task_id=task_id,
            project_id=project_id,
        )
        return {"success": False, "error": "Failed to generate any topics"}

    logger.info(
        "nightly_batch_topics_generated",
        task_id=task_id,
        project_id=project_id,
        topic_count=len(topics),
    )

    # Step 2: Plan batch (uses exploit/explore sampling)
    batch_result = plan_next_batch_task.apply(args=[project_id, n, topics]).get()

    if not batch_result.get("success"):
        logger.error(
            "nightly_batch_planning_failed",
            task_id=task_id,
            project_id=project_id,
            error=batch_result.get("error"),
        )
        return {
            "success": False,
            "error": f"Batch planning failed: {batch_result.get('error')}",
            "topics_generated": len(topics),
        }

    job_ids = batch_result.get("job_ids", [])
    logger.info(
        "nightly_batch_planned",
        task_id=task_id,
        project_id=project_id,
        batch_id=batch_result.get("batch_id"),
        jobs_created=len(job_ids),
    )

    # Step 3: Build optimization context for this project
    optimization_context_built = False
    with get_session_context() as session:
        try:
            context_builder = OptimizationContextBuilder(session)
            opt_context = context_builder.build(project_uuid)
            optimization_context_built = opt_context.sample_size >= 5
            logger.info(
                "optimization_context_built",
                project_id=project_id,
                sample_size=opt_context.sample_size,
                winning_patterns=len(opt_context.winning_patterns),
            )
        except Exception as e:
            logger.warning(
                "optimization_context_build_failed",
                project_id=project_id,
                error=str(e),
            )

    # Step 4: Dispatch video generation pipelines with auto-chain settings
    dispatched = []
    for job_id in job_ids:
        pipeline_result = run_pipeline_for_job_task.delay(
            job_id,
            auto_render=auto_render,
            auto_publish=auto_publish,
        )
        dispatched.append({"job_id": job_id, "task_id": pipeline_result.id})

    # Update batch status
    with get_session_context() as session:
        batch = session.get(PlannedBatchModel, UUID(batch_result["batch_id"]))
        if batch:
            batch.jobs_dispatched = len(dispatched)
            session.commit()

    logger.info(
        "nightly_batch_completed",
        task_id=task_id,
        project_id=project_id,
        batch_id=batch_result.get("batch_id"),
        topics_generated=len(topics),
        jobs_created=len(job_ids),
        jobs_dispatched=len(dispatched),
        auto_render=auto_render,
        auto_publish=auto_publish,
        optimization_context_built=optimization_context_built,
    )

    return {
        "success": True,
        "project_id": project_id,
        "batch_id": batch_result.get("batch_id"),
        "topics_generated": len(topics),
        "jobs_created": len(job_ids),
        "optimization_context_built": optimization_context_built,
        "jobs_dispatched": len(dispatched),
        "exploit_count": batch_result.get("exploit_count", 0),
        "explore_count": batch_result.get("explore_count", 0),
        "dispatched_tasks": dispatched,
    }


@celery_app.task(
    bind=True,
    name="autonomous.run_all_projects_nightly",
)
def run_all_projects_nightly_task(
    self: Any,
) -> dict[str, Any]:
    """Run nightly batch for all active projects.

    This is designed to be called from Celery beat scheduler.
    Iterates through all active projects and triggers nightly batch for each.

    Returns:
        Dict with results for each project
    """
    task_id = self.request.id
    settings = get_settings()

    if not settings.autonomous_enabled:
        logger.info(
            "all_projects_nightly_skipped",
            task_id=task_id,
            reason="autonomous_disabled",
        )
        return {
            "success": False,
            "error": "Autonomous mode is disabled",
        }

    logger.info("all_projects_nightly_started", task_id=task_id)

    with get_session_context() as session:
        from sqlalchemy import select

        projects = (
            session.execute(select(ProjectModel).where(ProjectModel.is_active.is_(True)))
            .scalars()
            .all()
        )

        project_ids = [str(p.id) for p in projects]

    if not project_ids:
        return {
            "success": True,
            "message": "No active projects found",
            "projects_processed": 0,
        }

    # Dispatch nightly batch for each project
    results = []
    for project_id in project_ids:
        result = run_nightly_batch_task.delay(project_id)
        results.append({"project_id": project_id, "task_id": result.id})

    logger.info(
        "all_projects_nightly_dispatched",
        task_id=task_id,
        projects_dispatched=len(results),
    )

    return {
        "success": True,
        "projects_dispatched": len(results),
        "tasks": results,
    }
