"""Focused tests for the manifest bridge exporter seam."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from shorts_engine.bridge.export_project_manifest import (
    ManifestExportResult,
    export_project_manifest,
)
from shorts_engine.db.models import AssetModel, ProjectModel, PromptModel, SceneModel, StoryModel, VideoJobModel
from shorts_engine.jobs.video_pipeline import _export_ready_manifest
from shorts_engine.services.storage import StorageService


class FakeScalarResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return self

    def all(self):
        return list(self._values)


class FakeSession:
    def __init__(self, job, execute_results=None):
        self.job = job
        self.execute_results = list(execute_results or [])
        self.commit_count = 0

    def get(self, model, key):  # noqa: ARG002
        if str(key) == str(self.job.id):
            return self.job
        return None

    def execute(self, _statement):
        if not self.execute_results:
            raise AssertionError("Unexpected execute call")
        return FakeScalarResult(self.execute_results.pop(0))

    def commit(self):
        self.commit_count += 1


def _build_job_fixture():
    project = ProjectModel(
        id=uuid4(),
        name="Bridge Test Project",
        description="Manifest export test project",
        default_style_preset="CINEMATIC_REALISM",
        settings={"captions": True, "client_id": "client-acme", "lane_id": "shortform_v1"},
    )
    story = StoryModel(
        id=uuid4(),
        project_id=project.id,
        topic="Bridge topic",
        title="Bridge title",
        narrative_text="A complete narrative.",
        narrative_style="documentary",
        suggested_preset="CINEMATIC_REALISM",
        word_count=42,
        estimated_duration_seconds=24.0,
        status="approved",
    )
    job = VideoJobModel(
        id=uuid4(),
        project_id=project.id,
        idempotency_key="idem-123",
        idea="Original idea",
        style_preset="CINEMATIC_REALISM",
        title="Bridge job",
        description="Bridge job description",
        status="completed",
        stage="ready",
        metadata_={"existing_key": "existing_value"},
        created_at=datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        completed_at=datetime(2025, 1, 2, 3, 14, 5, tzinfo=UTC),
    )
    job.project = project
    job.story = story

    scene_two = SceneModel(
        id=uuid4(),
        video_job_id=job.id,
        scene_number=2,
        visual_prompt="Fallback prompt two",
        continuity_notes="Continue the action",
        caption_beat="Second beat",
        duration_seconds=6.0,
        status="ready",
        metadata_={"shot": "medium"},
    )
    scene_one = SceneModel(
        id=uuid4(),
        video_job_id=job.id,
        scene_number=1,
        visual_prompt="Fallback prompt one",
        continuity_notes="Open on the hero",
        caption_beat="First beat",
        duration_seconds=5.0,
        status="ready",
        metadata_={"shot": "wide"},
    )
    prompt_one = PromptModel(
        id=uuid4(),
        scene_id=scene_one.id,
        prompt_type="visual",
        prompt_text="Final prompt one",
        model="planner-v1",
        version=1,
        is_final=True,
    )
    prompt_two = PromptModel(
        id=uuid4(),
        scene_id=scene_two.id,
        prompt_type="visual",
        prompt_text="Final prompt two",
        model="planner-v1",
        version=1,
        is_final=True,
    )
    asset = AssetModel(
        id=uuid4(),
        video_job_id=job.id,
        scene_id=scene_one.id,
        asset_type="scene_clip",
        storage_type="local",
        file_path="storage/clips/scene-one.mp4",
        url=None,
        provider="stub",
        duration_seconds=5.0,
        mime_type="video/mp4",
        width=1080,
        height=1920,
        status="ready",
        metadata_={"origin": "stub"},
    )
    return job, [scene_two, scene_one], [asset], [prompt_two, prompt_one]


def test_export_project_manifest_writes_expected_bundle(tmp_path):
    job, scenes, assets, prompts = _build_job_fixture()
    session = FakeSession(job, execute_results=[scenes, assets, prompts])
    storage = StorageService(base_path=tmp_path / "storage")

    result = export_project_manifest(session, job.id, storage_service=storage)

    manifest_file = tmp_path / "storage" / "manifests" / str(job.id) / "manifest.json"
    job_artifact_file = tmp_path / "storage" / "manifests" / str(job.id) / "shortform_job.v1.json"
    assert manifest_file.exists()
    assert job_artifact_file.exists()
    assert result.manifest_path == f"storage/manifests/{job.id}/manifest.json"
    assert result.job_artifact_path == f"storage/manifests/{job.id}/shortform_job.v1.json"

    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "manifest.v1"
    assert payload["project"]["id"] == str(job.project.id)
    assert payload["job"]["idempotency_key"] == "idem-123"
    assert payload["job"]["metadata_summary"]["keys"] == ["existing_key"]
    assert payload["story"]["title"] == "Bridge title"
    assert payload["preset"]["name"] == "CINEMATIC_REALISM"
    assert [scene["scene_number"] for scene in payload["scenes"]] == [1, 2]
    assert payload["scenes"][0]["prompt"] == "Final prompt one"
    assert payload["assets"][0]["local_file_path"] == "storage/clips/scene-one.mp4"
    assert payload["bundle"]["manifest_path"] == f"storage/manifests/{job.id}/manifest.json"

    job_artifact = json.loads(job_artifact_file.read_text(encoding="utf-8"))
    assert job_artifact["schema_version"] == "shortform_ai_generated_job.v1"
    assert job_artifact["job_id"] == str(job.id)
    assert job_artifact["client_id"] == "client-acme"
    assert job_artifact["lane_id"] == "shortform_v1"
    assert job_artifact["entry_path"] == "ai_generated"
    assert job_artifact["idea"] == "Original idea"
    assert job_artifact["script"] == "A complete narrative."
    assert job_artifact["project_manifest_ref"] == f"storage/manifests/{job.id}/manifest.json"
    assert len(job_artifact["preview_artifacts"]) == 1
    assert job_artifact["preview_artifacts"][0]["artifact_id"] == str(assets[0].id)
    assert job_artifact["asset_refs"][0]["asset_id"] == str(assets[0].id)


def test_export_ready_manifest_persists_bridge_metadata(monkeypatch):
    job = SimpleNamespace(id=uuid4(), metadata_={"existing": "value"})
    session = FakeSession(job)

    monkeypatch.setattr(
        "shorts_engine.jobs.video_pipeline.export_project_manifest",
        lambda _session, _job_id: ManifestExportResult(
            manifest_path=f"storage/manifests/{job.id}/manifest.json",
            job_artifact_path=f"storage/manifests/{job.id}/shortform_job.v1.json",
            bundle_root=f"storage/manifests/{job.id}",
            summary={"schema_version": "manifest.v1", "scene_count": 2, "asset_count": 1},
        ),
    )

    _export_ready_manifest(session, job, str(job.id))

    assert session.commit_count == 1
    assert job.metadata_["existing"] == "value"
    assert job.metadata_["manifest_bridge"]["status"] == "exported"
    assert job.metadata_["manifest_bridge"]["manifest_path"] == (
        f"storage/manifests/{job.id}/manifest.json"
    )
    assert job.metadata_["manifest_bridge"]["job_artifact_path"] == (
        f"storage/manifests/{job.id}/shortform_job.v1.json"
    )


def test_export_ready_manifest_records_failure_and_raises(monkeypatch):
    job = SimpleNamespace(id=uuid4(), metadata_={})
    session = FakeSession(job)

    def _raise(_session, _job_id):
        raise ValueError("bridge export blew up")

    monkeypatch.setattr("shorts_engine.jobs.video_pipeline.export_project_manifest", _raise)

    with pytest.raises(RuntimeError, match="Manifest export failed"):
        _export_ready_manifest(session, job, str(job.id))

    assert session.commit_count == 1
    assert job.metadata_["manifest_bridge"]["status"] == "failed"
    assert "bridge export blew up" in job.metadata_["manifest_bridge"]["error"]
