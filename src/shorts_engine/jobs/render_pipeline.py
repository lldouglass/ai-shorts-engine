"""Video render pipeline Celery tasks.

Pipeline stages:
1. generate_voiceover (optional) - Create narration audio from script
2. render_final_video - Compose final video with Creatomate
3. mark_job_ready_to_publish - Update job status and store final URL

All tasks support:
- Retries with exponential backoff
- Idempotency via database checks
- Proper error handling and status updates
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from celery import chain
from sqlalchemy import select

from shorts_engine.adapters.renderer.creatomate import (
    CreatomateProvider,
    CreatomateRenderRequest,
    SceneClip,
)
from shorts_engine.adapters.renderer.stub import StubRendererProvider
from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider
from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider
from shorts_engine.adapters.voiceover.stub import StubVoiceoverProvider
from shorts_engine.config import settings
from shorts_engine.db.models import AssetModel, SceneModel, VideoJobModel
from shorts_engine.db.session import get_session_context
from shorts_engine.logging import get_logger
from shorts_engine.services.storage import StorageService
from shorts_engine.utils import run_async
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def get_voiceover_provider():
    """Get the configured voiceover provider."""
    provider = getattr(settings, "voiceover_provider", "stub").lower()

    if provider == "elevenlabs":
        return ElevenLabsProvider()
    elif provider == "edge_tts":
        return EdgeTTSProvider()
    else:
        return StubVoiceoverProvider()


def get_renderer_provider():
    """Get the configured renderer provider."""
    provider = getattr(settings, "renderer_provider", "stub").lower()

    if provider == "creatomate":
        return CreatomateProvider()
    else:
        return StubRendererProvider()


# =============================================================================
# RENDER STAGE 1: Generate Voiceover (Optional)
# =============================================================================


@celery_app.task(
    bind=True,
    name="render.generate_voiceover",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
)
def generate_voiceover_task(
    self,
    video_job_id: str,
    narration_script: str | None = None,
    voice_id: str | None = None,
) -> dict[str, Any]:
    """Generate voiceover audio for the video.

    Args:
        video_job_id: UUID of the video job
        narration_script: Optional custom narration script (if None, uses caption beats)
        voice_id: Optional voice ID for the voiceover

    Returns:
        Dict with voiceover asset information
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info("generate_voiceover_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Check for existing voiceover asset (idempotency)
        existing_asset = session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == job_uuid,
                AssetModel.asset_type == "voiceover",
                AssetModel.status == "ready",
            )
        ).scalar_one_or_none()

        if existing_asset:
            logger.info("generate_voiceover_already_done", video_job_id=video_job_id)
            return {
                "success": True,
                "video_job_id": video_job_id,
                "asset_id": str(existing_asset.id),
                "url": existing_asset.url,
                "skipped": True,
            }

        # Build narration script from caption beats if not provided
        if not narration_script:
            scenes = session.execute(
                select(SceneModel)
                .where(SceneModel.video_job_id == job_uuid)
                .order_by(SceneModel.scene_number)
            ).scalars().all()

            if not scenes:
                return {
                    "success": False,
                    "video_job_id": video_job_id,
                    "error": "No scenes found for voiceover generation",
                }

            # Combine caption beats into narration
            narration_parts = []
            for scene in scenes:
                if scene.caption_beat:
                    narration_parts.append(scene.caption_beat)

            narration_script = ". ".join(narration_parts)

        if not narration_script.strip():
            logger.info("generate_voiceover_no_content", video_job_id=video_job_id)
            return {
                "success": True,
                "video_job_id": video_job_id,
                "skipped": True,
                "reason": "No narration content",
            }

        try:
            from shorts_engine.adapters.voiceover.base import VoiceoverRequest

            provider = get_voiceover_provider()
            request = VoiceoverRequest(
                text=narration_script,
                voice_id=voice_id or "narrator",
            )

            result = run_async(provider.generate(request))

            if not result.success:
                raise RuntimeError(f"Voiceover generation failed: {result.error_message}")

            # Store the voiceover asset
            storage = StorageService()
            stored = run_async(
                storage.store_bytes(
                    data=result.audio_data,
                    asset_type="voiceover",
                    video_job_id=job_uuid,
                    metadata={
                        "provider": provider.name,
                        "voice_id": voice_id,
                        "text_length": len(narration_script),
                    },
                )
            )

            # Create asset record
            asset = AssetModel(
                id=stored.id,
                video_job_id=job_uuid,
                asset_type="voiceover",
                storage_type=stored.storage_type,
                file_path=str(stored.file_path) if stored.file_path else None,
                url=stored.url,
                provider=provider.name,
                file_size_bytes=stored.file_size_bytes,
                duration_seconds=result.duration_seconds,
                mime_type="audio/mpeg",
                status="ready",
                metadata_=result.metadata,
            )
            session.add(asset)
            session.commit()

            logger.info(
                "generate_voiceover_completed",
                video_job_id=video_job_id,
                asset_id=str(asset.id),
                duration=result.duration_seconds,
            )

            return {
                "success": True,
                "video_job_id": video_job_id,
                "asset_id": str(asset.id),
                "url": stored.url or f"file://{stored.file_path}",
                "duration_seconds": result.duration_seconds,
            }

        except Exception as e:
            logger.error("generate_voiceover_error", error=str(e))
            raise


# =============================================================================
# RENDER STAGE 2: Render Final Video
# =============================================================================


@celery_app.task(
    bind=True,
    name="render.render_final_video",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
)
def render_final_video_task(
    self,
    video_job_id: str,
    voiceover_result: dict[str, Any] | None = None,
    include_captions: bool = True,
    background_music_url: str | None = None,
) -> dict[str, Any]:
    """Render the final video composition.

    Args:
        video_job_id: UUID of the video job
        voiceover_result: Optional result from voiceover generation
        include_captions: Whether to burn in captions
        background_music_url: Optional background music URL

    Returns:
        Dict with render information
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info("render_final_video_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Check for existing final video asset (idempotency)
        existing_asset = session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == job_uuid,
                AssetModel.asset_type == "final_video",
                AssetModel.status == "ready",
            )
        ).scalar_one_or_none()

        if existing_asset:
            logger.info("render_final_video_already_done", video_job_id=video_job_id)
            return {
                "success": True,
                "video_job_id": video_job_id,
                "asset_id": str(existing_asset.id),
                "url": existing_asset.url,
                "skipped": True,
            }

        # Update job stage
        job.stage = "rendering"
        session.commit()

        # Gather scene clips
        scenes = session.execute(
            select(SceneModel)
            .where(SceneModel.video_job_id == job_uuid)
            .order_by(SceneModel.scene_number)
        ).scalars().all()

        if not scenes:
            raise ValueError("No scenes found for rendering")

        # Build scene clips list
        scene_clips: list[SceneClip] = []

        for scene in scenes:
            # Get the scene's video clip asset
            clip_asset = session.execute(
                select(AssetModel).where(
                    AssetModel.scene_id == scene.id,
                    AssetModel.asset_type == "scene_clip",
                    AssetModel.status == "ready",
                )
            ).scalar_one_or_none()

            if not clip_asset:
                raise ValueError(f"No video clip found for scene {scene.scene_number}")

            # Get URL for the clip
            clip_url = clip_asset.url
            if not clip_url and clip_asset.file_path:
                clip_url = f"file://{clip_asset.file_path}"

            if not clip_url:
                raise ValueError(f"No URL available for scene {scene.scene_number} clip")

            scene_clips.append(SceneClip(
                video_url=clip_url,
                duration_seconds=scene.duration_seconds,
                caption_text=scene.caption_beat if include_captions else None,
                scene_number=scene.scene_number,
            ))

        # Get voiceover URL if available
        voiceover_url = None
        if voiceover_result and voiceover_result.get("success") and not voiceover_result.get("skipped"):
            voiceover_url = voiceover_result.get("url")

        # If no voiceover from result, check for existing asset
        if not voiceover_url:
            voiceover_asset = session.execute(
                select(AssetModel).where(
                    AssetModel.video_job_id == job_uuid,
                    AssetModel.asset_type == "voiceover",
                    AssetModel.status == "ready",
                )
            ).scalar_one_or_none()

            if voiceover_asset:
                voiceover_url = voiceover_asset.url or f"file://{voiceover_asset.file_path}"

        try:
            provider = get_renderer_provider()

            if isinstance(provider, CreatomateProvider):
                # Use Creatomate's composition rendering
                request = CreatomateRenderRequest(
                    scenes=scene_clips,
                    voiceover_url=voiceover_url,
                    background_music_url=background_music_url,
                    width=1080,
                    height=1920,
                    fps=30,
                )

                result = run_async(provider.render_composition(request))
            else:
                # Stub provider - simulate rendering
                from shorts_engine.adapters.renderer.base import RenderRequest

                request = RenderRequest(
                    video_data=b"",  # Stub doesn't use this
                    resolution="1080x1920",
                )
                result = run_async(provider.render(request))

            if not result.success:
                job.stage = "render_failed"
                job.error_message = result.error_message
                session.commit()
                raise RuntimeError(f"Render failed: {result.error_message}")

            # Get the output URL
            output_url = result.metadata.get("url") if result.metadata else None
            if not output_url and result.output_path:
                output_url = f"file://{result.output_path}"

            # Store the final video asset
            storage = StorageService()

            if output_url and output_url.startswith("http"):
                # Download and store from URL
                stored = run_async(
                    storage.store_from_url(
                        url=output_url,
                        asset_type="final_video",
                        video_job_id=job_uuid,
                        metadata={
                            "provider": provider.name,
                            "render_id": result.metadata.get("render_id") if result.metadata else None,
                        },
                    )
                )
            else:
                # URL reference only
                stored = storage.store_url_reference(
                    url=output_url or "stub://rendered",
                    asset_type="final_video",
                    video_job_id=job_uuid,
                    metadata=result.metadata or {},
                )

            # Create asset record
            asset = AssetModel(
                id=stored.id,
                video_job_id=job_uuid,
                asset_type="final_video",
                storage_type=stored.storage_type,
                file_path=str(stored.file_path) if stored.file_path else None,
                url=output_url or stored.url,
                provider=provider.name,
                file_size_bytes=result.file_size_bytes or stored.file_size_bytes,
                duration_seconds=result.duration_seconds,
                mime_type="video/mp4",
                width=1080,
                height=1920,
                status="ready",
                metadata_=result.metadata,
            )
            session.add(asset)

            # Update job with final video info
            job.stage = "rendered"
            session.commit()

            logger.info(
                "render_final_video_completed",
                video_job_id=video_job_id,
                asset_id=str(asset.id),
                url=output_url,
            )

            return {
                "success": True,
                "video_job_id": video_job_id,
                "asset_id": str(asset.id),
                "url": output_url,
                "duration_seconds": result.duration_seconds,
            }

        except Exception as e:
            job.stage = "render_failed"
            job.error_message = str(e)
            session.commit()
            raise


# =============================================================================
# RENDER STAGE 3: Mark Ready to Publish
# =============================================================================


@celery_app.task(
    bind=True,
    name="render.mark_ready_to_publish",
)
def mark_ready_to_publish_task(
    self,
    video_job_id: str,
    render_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark a video job as ready to publish.

    Args:
        video_job_id: UUID of the video job
        render_result: Optional result from render stage

    Returns:
        Dict with final status and MP4 URL
    """
    task_id = self.request.id
    job_uuid = UUID(video_job_id)

    logger.info("mark_ready_to_publish_started", task_id=task_id, video_job_id=video_job_id)

    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            raise ValueError(f"Video job not found: {video_job_id}")

        # Check render result
        if render_result and not render_result.get("success"):
            job.status = "failed"
            job.stage = "render_failed"
            job.error_message = render_result.get("error", "Render failed")
            session.commit()

            return {
                "success": False,
                "video_job_id": video_job_id,
                "error": job.error_message,
            }

        # Get the final video asset
        final_asset = session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == job_uuid,
                AssetModel.asset_type == "final_video",
                AssetModel.status == "ready",
            )
        ).scalar_one_or_none()

        if not final_asset:
            job.status = "failed"
            job.stage = "render_failed"
            job.error_message = "No final video asset found"
            session.commit()

            return {
                "success": False,
                "video_job_id": video_job_id,
                "error": "No final video asset found",
            }

        # Get final MP4 URL
        final_mp4_url = final_asset.url
        if not final_mp4_url and final_asset.file_path:
            final_mp4_url = f"file://{final_asset.file_path}"

        # Update job status
        job.status = "completed"
        job.stage = "ready_to_publish"
        job.completed_at = datetime.now(timezone.utc)

        # Store final URL in metadata
        job.metadata_ = job.metadata_ or {}
        job.metadata_["final_mp4_url"] = final_mp4_url
        job.metadata_["final_asset_id"] = str(final_asset.id)

        session.commit()

        logger.info(
            "mark_ready_to_publish_completed",
            video_job_id=video_job_id,
            final_mp4_url=final_mp4_url,
        )

        return {
            "success": True,
            "video_job_id": video_job_id,
            "stage": "ready_to_publish",
            "title": job.title,
            "description": job.description,
            "final_mp4_url": final_mp4_url,
            "final_asset_id": str(final_asset.id),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }


# =============================================================================
# FULL RENDER PIPELINE ORCHESTRATOR
# =============================================================================


@celery_app.task(
    bind=True,
    name="render.run_render_pipeline",
)
def run_render_pipeline_task(
    self,
    video_job_id: str,
    include_voiceover: bool = True,
    include_captions: bool = True,
    voice_id: str | None = None,
    narration_script: str | None = None,
    background_music_url: str | None = None,
) -> dict[str, Any]:
    """Run the full render pipeline.

    Stages:
    1. Generate voiceover (if enabled)
    2. Render final video
    3. Mark ready to publish

    Args:
        video_job_id: UUID of the video job
        include_voiceover: Whether to generate voiceover
        include_captions: Whether to burn in captions
        voice_id: Optional voice ID for voiceover
        narration_script: Optional custom narration script
        background_music_url: Optional background music URL

    Returns:
        Dict with job ID and pipeline status
    """
    task_id = self.request.id

    logger.info(
        "render_pipeline_started",
        task_id=task_id,
        video_job_id=video_job_id,
        include_voiceover=include_voiceover,
    )

    # Build pipeline chain
    if include_voiceover:
        pipeline = chain(
            generate_voiceover_task.s(
                video_job_id,
                narration_script=narration_script,
                voice_id=voice_id,
            ),
            render_final_video_task.s(
                video_job_id=video_job_id,
                include_captions=include_captions,
                background_music_url=background_music_url,
            ),
            mark_ready_to_publish_task.s(video_job_id=video_job_id),
        )
    else:
        pipeline = chain(
            render_final_video_task.s(
                video_job_id=video_job_id,
                voiceover_result=None,
                include_captions=include_captions,
                background_music_url=background_music_url,
            ),
            mark_ready_to_publish_task.s(video_job_id=video_job_id),
        )

    # Start pipeline
    result = pipeline.apply_async()

    return {
        "success": True,
        "video_job_id": video_job_id,
        "pipeline_task_id": result.id,
        "stages": ["voiceover", "render", "ready_to_publish"] if include_voiceover else ["render", "ready_to_publish"],
    }
