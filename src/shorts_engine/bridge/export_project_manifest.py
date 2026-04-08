"""Export a pragmatic project manifest for TypeScript studio handoff."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select

from shorts_engine.db.models import AssetModel, PromptModel, SceneModel, VideoJobModel
from shorts_engine.presets.styles import get_preset
from shorts_engine.services.storage import StorageService

SCHEMA_VERSION = "manifest.v1"
SHORTFORM_JOB_SCHEMA_VERSION = "shortform_ai_generated_job.v1"
DEFAULT_LANE_ID = "shortform_v1"


@dataclass(frozen=True)
class ManifestExportResult:
    """Result metadata for a manifest export."""

    manifest_path: str
    job_artifact_path: str
    bundle_root: str
    summary: dict[str, Any]


def export_project_manifest(
    session: Any,
    video_job_id: UUID | str,
    storage_service: StorageService | None = None,
) -> ManifestExportResult:
    """Export a job manifest bundle to storage/manifests/<video_job_id>/manifest.json."""
    job_uuid = UUID(str(video_job_id))
    storage = storage_service or StorageService()

    job = session.get(VideoJobModel, job_uuid)
    if not job:
        raise ValueError(f"Video job not found: {video_job_id}")
    if not job.project:
        raise ValueError(f"Project not found for video job: {video_job_id}")

    scenes = (
        session.execute(
            select(SceneModel)
            .where(SceneModel.video_job_id == job_uuid)
            .order_by(SceneModel.scene_number)
        )
        .scalars()
        .all()
    )
    assets = (
        session.execute(
            select(AssetModel)
            .where(AssetModel.video_job_id == job_uuid)
            .order_by(AssetModel.created_at, AssetModel.id)
        )
        .scalars()
        .all()
    )
    prompts = (
        session.execute(
            select(PromptModel)
            .join(SceneModel, PromptModel.scene_id == SceneModel.id)
            .where(SceneModel.video_job_id == job_uuid)
            .order_by(SceneModel.scene_number, PromptModel.is_final.desc(), PromptModel.created_at)
        )
        .scalars()
        .all()
    )

    final_prompts_by_scene: dict[UUID, PromptModel] = {}
    for prompt in prompts:
        if prompt.scene_id not in final_prompts_by_scene or prompt.is_final:
            final_prompts_by_scene[prompt.scene_id] = prompt

    manifest_relative_path = Path("storage") / "manifests" / str(job.id) / "manifest.json"
    job_artifact_relative_path = (
        Path("storage") / "manifests" / str(job.id) / "shortform_job.v1.json"
    )
    bundle_relative_root = manifest_relative_path.parent
    bundle_dir = storage.base_path / "manifests" / str(job.id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / "manifest.json"
    job_artifact_path = bundle_dir / "shortform_job.v1.json"

    ordered_scenes = sorted(scenes, key=lambda scene: (scene.scene_number, str(scene.id)))
    ordered_assets = sorted(
        assets,
        key=lambda asset: (
            asset.created_at or datetime.min,
            str(asset.scene_id) if asset.scene_id else "",
            str(asset.id),
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "id": str(job.project.id),
            "name": job.project.name,
            "description": job.project.description,
            "default_style_preset": job.project.default_style_preset,
            "settings": job.project.settings or {},
        },
        "job": {
            "id": str(job.id),
            "idempotency_key": job.idempotency_key,
            "style_preset": job.style_preset,
            "title": job.title,
            "description": job.description,
            "stage": job.stage,
            "created_at": _isoformat(job.created_at),
            "completed_at": _isoformat(job.completed_at),
            "metadata_summary": _summarize_metadata(job.metadata_),
        },
        "story": _serialize_story(job.story),
        "preset": _serialize_preset(job.style_preset),
        "scenes": [
            {
                "id": str(scene.id),
                "scene_number": scene.scene_number,
                "prompt": final_prompts_by_scene.get(scene.id).prompt_text
                if final_prompts_by_scene.get(scene.id)
                else scene.visual_prompt,
                "continuity": scene.continuity_notes,
                "caption": scene.caption_beat,
                "duration_seconds": scene.duration_seconds,
                "status": scene.status,
                "metadata": scene.metadata_ or {},
            }
            for scene in ordered_scenes
        ],
        "assets": [
            {
                "id": str(asset.id),
                "scene_id": str(asset.scene_id) if asset.scene_id else None,
                "type": asset.asset_type,
                "local_file_path": asset.file_path,
                "url": asset.url,
                "provider": asset.provider,
                "duration_seconds": asset.duration_seconds,
                "mime_type": asset.mime_type,
                "width": asset.width,
                "height": asset.height,
                "status": asset.status,
                "metadata": asset.metadata_ or {},
            }
            for asset in ordered_assets
        ],
        "bundle": {
            "manifest_path": manifest_relative_path.as_posix(),
            "bundle_root": bundle_relative_root.as_posix(),
            "storage_root": storage.base_path.as_posix(),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    asset_refs = [_serialize_asset_ref(asset) for asset in ordered_assets]
    job_artifact = {
        "schema_version": SHORTFORM_JOB_SCHEMA_VERSION,
        "job_id": str(job.id),
        "client_id": _resolve_client_id(job),
        "lane_id": _resolve_lane_id(job),
        "entry_path": "ai_generated",
        "idea": job.idea,
        "script": _resolve_script(job),
        "preview_artifacts": [
            _serialize_preview_artifact(asset)
            for asset in ordered_assets
            if asset.asset_type == "scene_clip"
        ],
        "project_manifest_ref": manifest_relative_path.as_posix(),
        "asset_refs": asset_refs,
    }

    job_artifact_path.write_text(
        json.dumps(job_artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "job_artifact_schema_version": SHORTFORM_JOB_SCHEMA_VERSION,
        "scene_count": len(scenes),
        "asset_count": len(assets),
        "preview_artifact_count": len(job_artifact["preview_artifacts"]),
        "exported_at": datetime.now().isoformat(),
    }

    return ManifestExportResult(
        manifest_path=manifest_relative_path.as_posix(),
        job_artifact_path=job_artifact_relative_path.as_posix(),
        bundle_root=bundle_relative_root.as_posix(),
        summary=summary,
    )


def _serialize_story(story: Any) -> dict[str, Any] | None:
    if not story:
        return None
    return {
        "id": str(story.id),
        "topic": story.topic,
        "title": story.title,
        "narrative_text": story.narrative_text,
        "narrative_style": story.narrative_style,
        "suggested_preset": story.suggested_preset,
        "word_count": story.word_count,
        "estimated_duration_seconds": story.estimated_duration_seconds,
    }


def _serialize_preset(preset_name: str) -> dict[str, Any] | None:
    preset = get_preset(preset_name)
    if not preset:
        return None

    preset_data = asdict(preset)
    aspect_ratio = preset_data.get("aspect_ratio")
    if aspect_ratio is not None:
        preset_data["aspect_ratio"] = str(aspect_ratio)
    return preset_data


def _summarize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        "keys": sorted(metadata.keys()),
        "final_asset_id": metadata.get("final_asset_id"),
        "final_mp4_url": metadata.get("final_mp4_url"),
    }


def _resolve_client_id(job: VideoJobModel) -> str:
    project_settings = job.project.settings or {}
    metadata = job.metadata_ or {}
    manifest_bridge = metadata.get("manifest_bridge", {})
    return str(
        project_settings.get("client_id")
        or metadata.get("client_id")
        or manifest_bridge.get("client_id")
        or job.project.id
    )


def _resolve_lane_id(job: VideoJobModel) -> str:
    project_settings = job.project.settings or {}
    metadata = job.metadata_ or {}
    manifest_bridge = metadata.get("manifest_bridge", {})
    return str(
        project_settings.get("lane_id")
        or metadata.get("lane_id")
        or manifest_bridge.get("lane_id")
        or DEFAULT_LANE_ID
    )


def _resolve_script(job: VideoJobModel) -> str:
    if job.story and job.story.narrative_text:
        return job.story.narrative_text
    if job.description:
        return job.description
    return job.idea


def _serialize_preview_artifact(asset: AssetModel) -> dict[str, Any]:
    return {
        "artifact_id": str(asset.id),
        "artifact_type": asset.asset_type,
        "scene_id": str(asset.scene_id) if asset.scene_id else None,
        "path": asset.file_path,
        "url": asset.url,
        "duration_seconds": asset.duration_seconds,
        "mime_type": asset.mime_type,
        "width": asset.width,
        "height": asset.height,
        "status": asset.status,
    }


def _serialize_asset_ref(asset: AssetModel) -> dict[str, Any]:
    return {
        "asset_id": str(asset.id),
        "scene_id": str(asset.scene_id) if asset.scene_id else None,
        "asset_type": asset.asset_type,
        "path": asset.file_path,
        "url": asset.url,
        "provider": asset.provider,
        "duration_seconds": asset.duration_seconds,
        "mime_type": asset.mime_type,
        "width": asset.width,
        "height": asset.height,
        "status": asset.status,
    }


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
