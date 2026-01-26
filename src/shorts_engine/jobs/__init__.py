"""Celery job definitions."""

from shorts_engine.jobs.tasks import (
    generate_video_task,
    ingest_analytics_task,
    ingest_comments_task,
    publish_video_task,
    render_video_task,
    smoke_test_task,
)
from shorts_engine.jobs.video_pipeline import (
    generate_all_scenes_task,
    generate_scene_clip_task,
    mark_ready_for_render_task,
    plan_job_task,
    run_full_pipeline_task,
    verify_assets_task,
)
from shorts_engine.jobs.render_pipeline import (
    generate_voiceover_task,
    mark_ready_to_publish_task,
    render_final_video_task,
    run_render_pipeline_task,
)
from shorts_engine.jobs.publish_pipeline import (
    check_publish_status_task,
    publish_to_youtube_task,
    run_publish_pipeline_task,
)

__all__ = [
    # Core tasks
    "generate_video_task",
    "ingest_analytics_task",
    "ingest_comments_task",
    "publish_video_task",
    "render_video_task",
    "smoke_test_task",
    # Video creation pipeline tasks
    "plan_job_task",
    "generate_scene_clip_task",
    "generate_all_scenes_task",
    "verify_assets_task",
    "mark_ready_for_render_task",
    "run_full_pipeline_task",
    # Render pipeline tasks
    "generate_voiceover_task",
    "render_final_video_task",
    "mark_ready_to_publish_task",
    "run_render_pipeline_task",
    # Publish pipeline tasks
    "publish_to_youtube_task",
    "run_publish_pipeline_task",
    "check_publish_status_task",
]
