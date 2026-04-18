"""Tests for deterministic shot-plan use inside the video pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from shorts_engine.db.models import PromptModel, SceneModel, VideoJobModel
from shorts_engine.domain.enums import QAStatus
from shorts_engine.jobs import video_pipeline
from shorts_engine.shot_plans import FLAGSHIP_PRESET_ID, FLAGSHIP_PRESET_VERSION


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
    assert len(plan.scenes) == 3

    first_scene = plan.scenes[0]
    assert first_scene.scene_number == 1
    assert "Aurum Glow Serum macro detail" in first_scene.visual_prompt
    assert "Avoid readable text" in first_scene.visual_prompt
    assert first_scene.metadata
    assert first_scene.metadata["shot_plan"]["plan_id"] == plan.raw_response["plan_id"]
    assert first_scene.metadata["shot_plan"]["take_request"]["shot_id"]


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

    assert len(fake_session.scenes) == 3
    assert len(fake_session.prompts) == 3
    assert fake_session.prompts[0].model == video_pipeline.SHOT_PLAN_PROMPT_MODEL
    assert fake_session.scenes[0].metadata_["shot_plan"]["plan_id"] == job.plan_data["plan_id"]
    assert fake_session.scenes[0].metadata_ == fake_session.prompts[0].metadata_


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
