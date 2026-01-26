"""Celery worker configuration."""

from celery import Celery

from shorts_engine.config import settings
from shorts_engine.logging import setup_logging

# Setup logging before anything else
setup_logging()

# Create Celery app
celery_app = Celery(
    "shorts_engine",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=600,  # 10 minutes max
    task_soft_time_limit=540,  # 9 minutes soft limit
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    # Result backend
    result_expires=86400,  # 24 hours
    # Task routing
    task_routes={
        "smoke_test": {"queue": "default"},
        "generate_video": {"queue": "high"},
        "render_video": {"queue": "high"},
        "publish_video": {"queue": "high"},
        "ingest_analytics": {"queue": "low"},
        "ingest_comments": {"queue": "low"},
        # Video pipeline tasks
        "pipeline.plan_job": {"queue": "high"},
        "pipeline.generate_scene_clip": {"queue": "high"},
        "pipeline.generate_all_scenes": {"queue": "high"},
        "pipeline.verify_assets": {"queue": "high"},
        "pipeline.mark_ready_for_render": {"queue": "high"},
        "pipeline.run_full_pipeline": {"queue": "high"},
        "pipeline.generate_all_scenes_from_plan": {"queue": "high"},
        # Render pipeline tasks
        "render.generate_voiceover": {"queue": "high"},
        "render.render_final_video": {"queue": "high"},
        "render.mark_ready_to_publish": {"queue": "high"},
        "render.run_render_pipeline": {"queue": "high"},
    },
    # Beat scheduler (for periodic tasks)
    beat_schedule={
        # Example: ingest analytics every hour
        # "ingest-analytics-hourly": {
        #     "task": "ingest_analytics",
        #     "schedule": 3600.0,
        #     "args": (),
        # },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["shorts_engine.jobs"])
