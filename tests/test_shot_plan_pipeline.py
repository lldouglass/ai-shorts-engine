"""Tests for deterministic shot-plan use inside the video pipeline."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from shorts_engine.adapters.video_gen.base import VideoGenProvider, VideoGenRequest, VideoGenResult
from shorts_engine.db.models import PromptModel, SceneModel, VideoJobModel
from shorts_engine.domain.enums import QAStatus
from shorts_engine.jobs import video_pipeline
from shorts_engine.services.ref_pack_generator import GeneratedReferenceImage, RefPackGenerator
from shorts_engine.services.storage import StorageService
from shorts_engine.shot_plans import FLAGSHIP_PRESET_ID, FLAGSHIP_PRESET_VERSION
from shorts_engine.utils import run_async

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlH0i8AAAAASUVORK5CYII="
)


class FakeReferenceImageGenerator:
    """Deterministic fake used to exercise the real ref-pack approval path."""

    async def generate(
        self,
        prompt: str,
        *,
        candidate_index: int,
        aspect_ratio: str,
        model: str,
        shot,
        reference_assets,
    ) -> GeneratedReferenceImage:
        del prompt, candidate_index, aspect_ratio, model, shot, reference_assets

        return GeneratedReferenceImage(
            image_bytes=PNG_BYTES,
            mime_type="image/png",
            provider_params={"generator": "fake"},
        )


class FakeVideoGenProvider(VideoGenProvider):
    def __init__(self, results: list[VideoGenResult], *, supports_reference_images: bool = True) -> None:
        self._results = results
        self._supports_reference_images = supports_reference_images
        self.requests: list[VideoGenRequest] = []

    @property
    def name(self) -> str:
        return "fake-video-provider"

    @property
    def supports_reference_images(self) -> bool:
        return self._supports_reference_images

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        self.requests.append(request)
        return self._results[len(self.requests) - 1]

    async def check_status(self, job_id: str) -> dict[str, Any]:
        return {"job_id": job_id, "status": "completed"}


def _product_input() -> dict[str, Any]:
    return {
        "product_name": "Aurum Glow Serum",
        "product_category": "premium skincare",
        "concept": "Aurum Glow Serum premium ad",
        "key_benefit": "a clean glass-skin glow in one drop",
        "audience": "skincare buyers who want a premium routine upgrade",
        "primary_sensory_cue": "golden serum bead texture on glass",
        "supporting_detail": "the glass dropper and luminous serum bead",
        "use_case": "morning skincare routine",
        "visual_constraints": ["no readable label text", "no medical claims"],
    }


def _brand_input() -> dict[str, Any]:
    return {
        "brand_name": "Aurum",
        "brand_voice": "premium, restrained, sensory",
        "visual_style": "cinematic commercial realism",
        "environment": "warm marble bathroom counter with soft window light",
    }


def test_build_video_plan_from_shot_plan_feeds_generation_contract() -> None:
    """Shot-plan selectors compile into the VideoPlan shape generation already uses."""
    style_preset = f"{video_pipeline.SHOT_PLAN_STYLE_PREFIX}{FLAGSHIP_PRESET_ID}"

    assert video_pipeline.resolve_shot_plan_preset("DARK_DYSTOPIAN_ANIME") is None

    plan = video_pipeline.build_video_plan_from_shot_plan(
        "Aurum Glow Serum premium ad",
        style_preset,
        product=_product_input(),
        brand=_brand_input(),
    )

    assert plan is not None
    assert plan.style_preset == style_preset
    assert plan.raw_response
    assert plan.raw_response["preset"]["preset_id"] == FLAGSHIP_PRESET_ID
    assert plan.raw_response["preset"]["version"] == FLAGSHIP_PRESET_VERSION
    assert len(plan.scenes) == 6

    first_scene = plan.scenes[0]
    assert first_scene.scene_number == 1
    assert "Aurum Glow Serum macro detail" in first_scene.visual_prompt
    assert "Avoid readable text" in first_scene.visual_prompt
    assert "Sequence continuity locks:" in first_scene.continuity_notes
    assert "Aurum Glow Serum" in first_scene.continuity_notes
    assert first_scene.metadata
    assert first_scene.metadata["shot_plan"]["plan_id"] == plan.raw_response["plan_id"]
    assert first_scene.metadata["shot_plan"]["shot"]["storyboard_board"]["title"] == "Hook"
    assert (
        first_scene.metadata["shot_plan"]["shot"]["storyboard_board"]["on_frame_text"]
        == "a clean glass-skin glow in one drop"
    )
    assert first_scene.metadata["shot_plan"]["take_request"]["shot_id"]
    assert first_scene.metadata["shot_plan"]["take_request"]["storyboard_deck"]["layout_system"]


def test_plan_job_task_compiles_shot_plan_into_generation_rows(monkeypatch) -> None:
    """The real planning task persists compiled shot-plan scenes for generation."""
    job_id = uuid4()
    job = SimpleNamespace(
        id=job_id,
        project_id=uuid4(),
        idea="Aurum Glow Serum premium ad",
        style_preset=f"{video_pipeline.SHOT_PLAN_STYLE_PREFIX}{FLAGSHIP_PRESET_ID}",
        metadata_={"shot_plan": {"product": _product_input(), "brand": _brand_input()}},
        title=None,
        description=None,
        plan_data=None,
        scenes=[],
        status="pending",
        stage="created",
        celery_task_id=None,
        started_at=None,
        story_id=None,
        story=None,
        qa_status=None,
        qa_attempts=0,
        last_qa_error=None,
        error_message=None,
        retry_count=0,
    )
    fake_session = _FakePlanSession(job)

    @contextmanager
    def fake_session_context():
        yield fake_session

    async def fail_llm_plan(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("LLM planner should not run for shot-plan presets")

    monkeypatch.setattr(video_pipeline, "get_session_context", fake_session_context)
    monkeypatch.setattr(video_pipeline.settings, "qa_enabled", False)
    monkeypatch.setattr(video_pipeline, "record_job_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_pipeline, "record_job_completed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_pipeline, "record_cost_estimate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_pipeline.PlannerService, "plan", fail_llm_plan)

    result = video_pipeline.plan_job_task.run(str(job_id))

    assert result["success"] is True
    assert result["scene_ids"] == [str(scene.id) for scene in fake_session.scenes]
    assert result["total_duration"] == 8.0
    assert job.stage == "planned"
    assert job.qa_status == QAStatus.SKIPPED
    assert job.plan_data["preset"]["preset_id"] == FLAGSHIP_PRESET_ID

    assert len(fake_session.scenes) == 6
    assert len(fake_session.prompts) == 6
    assert fake_session.prompts[0].model == video_pipeline.SHOT_PLAN_PROMPT_MODEL
    assert fake_session.scenes[0].metadata_["shot_plan"]["plan_id"] == job.plan_data["plan_id"]
    assert fake_session.scenes[0].metadata_["shot_plan"]["shot"]["storyboard_board"]["title"] == "Hook"
    assert "Sequence continuity locks:" in fake_session.scenes[0].continuity_notes
    assert fake_session.scenes[0].metadata_ == fake_session.prompts[0].metadata_


@pytest.mark.asyncio
async def test_plan_job_task_persists_approved_board_lineage_in_scene_take_requests(
    monkeypatch,
    tmp_path,
) -> None:
    """Approved ref-pack selections flow into the normal plan-job take_request payload."""
    job_id = uuid4()
    product = _product_input()
    brand = _brand_input()
    compiled = video_pipeline.PlannerService.compile_shot_plan(
        product=product,
        brand=brand,
        preset_id=FLAGSHIP_PRESET_ID,
        preset_version=FLAGSHIP_PRESET_VERSION,
    )
    created_at = datetime(2026, 4, 23, 20, 15, tzinfo=UTC)
    ref_pack = await RefPackGenerator(
        image_generator=FakeReferenceImageGenerator(),
        output_root=tmp_path / "references",
        now_factory=lambda: created_at,
    ).generate(
        job_id="job-shot-plan-approval-pipeline-001",
        shot_plan=compiled,
        candidates_per_shot=2,
    )
    approved_candidate = ref_pack.shots[0].reference_candidates[0]
    job = SimpleNamespace(
        id=job_id,
        project_id=uuid4(),
        idea="Aurum Glow Serum premium ad",
        style_preset=f"{video_pipeline.SHOT_PLAN_STYLE_PREFIX}{FLAGSHIP_PRESET_ID}",
        metadata_={
            "shot_plan": {
                "product": product,
                "brand": brand,
                "ref_pack": ref_pack.model_dump(mode="json"),
                "approved_ref_ids_by_shot": {
                    compiled.shots[0].shot_id: approved_candidate.ref_id,
                },
            }
        },
        title=None,
        description=None,
        plan_data=None,
        scenes=[],
        status="pending",
        stage="created",
        celery_task_id=None,
        started_at=None,
        story_id=None,
        story=None,
        qa_status=None,
        qa_attempts=0,
        last_qa_error=None,
        error_message=None,
        retry_count=0,
    )
    fake_session = _FakePlanSession(job)

    @contextmanager
    def fake_session_context():
        yield fake_session

    async def fail_llm_plan(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("LLM planner should not run for shot-plan presets")

    monkeypatch.setattr(video_pipeline, "get_session_context", fake_session_context)
    monkeypatch.setattr(video_pipeline.settings, "qa_enabled", False)
    monkeypatch.setattr(video_pipeline, "record_job_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_pipeline, "record_job_completed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_pipeline, "record_cost_estimate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_pipeline.PlannerService, "plan", fail_llm_plan)

    result = video_pipeline.plan_job_task.run(str(job_id))

    assert result["success"] is True
    first_take_request = fake_session.scenes[0].metadata_["shot_plan"]["take_request"]
    second_take_request = fake_session.scenes[1].metadata_["shot_plan"]["take_request"]
    assert job.plan_data["status"] == "needs_references"
    assert job.plan_data["shots"][0]["take_request"]["approved_board"]["ref_id"] == approved_candidate.ref_id
    assert first_take_request["status"] == "ready"
    assert first_take_request["approved_board"] == {
        "ref_id": approved_candidate.ref_id,
        "asset_path": approved_candidate.asset_path,
        "source_ref_pack_id": ref_pack.ref_pack_id,
        "source_review_payload_id": ref_pack.lineage.source_review_payload_id,
        "first_frame_prompt_id": approved_candidate.params.first_frame_prompt_id,
    }
    assert second_take_request["approved_board"] is None


def test_generate_single_scene_clip_uses_approved_board_reference_flow(
    monkeypatch,
    tmp_path,
) -> None:
    """Shot-plan scene generation must animate from the approved board instead of freehanding."""
    job_id = uuid4()
    scene_id = uuid4()
    product = _product_input()
    brand = _brand_input()
    compiled = video_pipeline.PlannerService.compile_shot_plan(
        product=product,
        brand=brand,
        preset_id=FLAGSHIP_PRESET_ID,
        preset_version=FLAGSHIP_PRESET_VERSION,
    )
    created_at = datetime(2026, 4, 23, 20, 15, tzinfo=UTC)
    ref_pack = run_async(
        RefPackGenerator(
            image_generator=FakeReferenceImageGenerator(),
            output_root=tmp_path / "references",
            now_factory=lambda: created_at,
        ).generate(
            job_id="job-shot-plan-scene-generate-001",
            shot_plan=compiled,
            candidates_per_shot=2,
        )
    )
    approved_candidate = ref_pack.shots[0].reference_candidates[0]
    video_plan = video_pipeline.build_video_plan_from_shot_plan(
        "Aurum Glow Serum premium ad",
        f"{video_pipeline.SHOT_PLAN_STYLE_PREFIX}{FLAGSHIP_PRESET_ID}",
        product=product,
        brand=brand,
        ref_pack=ref_pack.model_dump(mode="json"),
        approved_ref_ids_by_shot={compiled.shots[0].shot_id: approved_candidate.ref_id},
    )

    assert video_plan is not None

    scene = SimpleNamespace(
        id=scene_id,
        video_job_id=job_id,
        scene_number=1,
        visual_prompt=video_plan.scenes[0].visual_prompt,
        continuity_notes=video_plan.scenes[0].continuity_notes,
        duration_seconds=video_plan.scenes[0].duration_seconds,
        status="pending",
        generation_attempts=0,
        last_error=None,
        metadata_=video_plan.scenes[0].metadata,
    )
    job = SimpleNamespace(
        id=job_id,
        style_preset=f"{video_pipeline.SHOT_PLAN_STYLE_PREFIX}{FLAGSHIP_PRESET_ID}",
    )
    fake_session = _FakeGenerateSceneSession(job=job, scene=scene)
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"SHOT_PLAN_SCENE_CLIP",
                metadata={
                    "generation_id": "gen-shot-plan-scene-01",
                    "model": "fake-video-model",
                },
            )
        ]
    )

    @contextmanager
    def fake_session_context():
        yield fake_session

    class TestStorageService(StorageService):
        def __init__(self) -> None:
            super().__init__(base_path=tmp_path / "storage")

    monkeypatch.setattr(video_pipeline, "get_session_context", fake_session_context)
    monkeypatch.setattr(video_pipeline, "get_video_gen_provider", lambda: provider)
    monkeypatch.setattr(video_pipeline, "StorageService", TestStorageService)

    result = video_pipeline._generate_single_scene_clip(str(scene_id), str(job_id))

    assert result["success"] is True
    assert len(provider.requests) == 1
    assert provider.requests[0].reference_images is not None
    assert len(provider.requests[0].reference_images) == 1
    assert "approved storyboard board / first-frame direction" in provider.requests[0].prompt
    assert scene.status == "generated"
    assert scene.last_error is None
    assert len(fake_session.assets) == 1

    asset = fake_session.assets[0]
    assert asset.file_path == result["file_path"]
    assert asset.file_path is not None
    assert asset.metadata_["shot_take"]["lineage"]["approved_board_ref_id"] == approved_candidate.ref_id
    assert (
        asset.metadata_["shot_take"]["lineage"]["approved_board_asset_path"]
        == approved_candidate.asset_path
    )
    assert asset.metadata_["shot_take_batch_id"]
    assert asset.external_id == "gen-shot-plan-scene-01"


class _FakePlanSession:
    def __init__(self, job: SimpleNamespace) -> None:
        self.job = job
        self.scenes: list[SceneModel] = []
        self.prompts: list[PromptModel] = []
        self.commits = 0

    def get(self, model: type[Any], item_id: Any) -> Any:
        if model is VideoJobModel and item_id == self.job.id:
            return self.job
        return None

    def add(self, item: Any) -> None:
        if isinstance(item, SceneModel):
            self.scenes.append(item)
            self.job.scenes.append(item)
        elif isinstance(item, PromptModel):
            self.prompts.append(item)

    def delete(self, item: Any) -> None:
        if item in self.job.scenes:
            self.job.scenes.remove(item)
        if item in self.scenes:
            self.scenes.remove(item)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.commits += 1


class _FakeScalarResult:
    def scalar_one_or_none(self) -> None:
        return None


class _FakeGenerateSceneSession:
    def __init__(self, *, job: SimpleNamespace, scene: SimpleNamespace) -> None:
        self.job = job
        self.scene = scene
        self.assets: list[Any] = []
        self.commits = 0

    def get(self, model: type[Any], item_id: Any) -> Any:
        if model is VideoJobModel and item_id == self.job.id:
            return self.job
        if model is SceneModel and item_id == self.scene.id:
            return self.scene
        return None

    def execute(self, _query: Any) -> _FakeScalarResult:
        return _FakeScalarResult()

    def add(self, item: Any) -> None:
        self.assets.append(item)

    def commit(self) -> None:
        self.commits += 1
