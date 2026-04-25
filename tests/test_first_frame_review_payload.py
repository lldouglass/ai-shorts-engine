"""Tests for first-frame review payload exports."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from shorts_engine.cli import app
from shorts_engine.shot_plans import (
    FirstFrameReviewPayload,
    FirstFrameReviewStatus,
    build_first_frame_review_payload,
)
from shorts_engine.shot_plans.benchmarks import (
    TALL_OWL_REFERENCE_ASSETS,
    build_tall_owl_first_frame_review_payload,
    compile_tall_owl_benchmark_shot_plan,
)


def test_first_frame_review_payload_builds_from_compiled_plan() -> None:
    """The review payload is a deterministic export built from CompiledShotPlan."""
    plan = compile_tall_owl_benchmark_shot_plan()
    payload = build_first_frame_review_payload(
        plan,
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
    )
    second = build_first_frame_review_payload(
        plan,
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
    )
    round_tripped = FirstFrameReviewPayload.model_validate_json(payload.model_dump_json())

    assert round_tripped == payload
    assert payload.payload_id == second.payload_id
    assert payload.model_dump(mode="json") == second.model_dump(mode="json")
    assert payload.source_plan_id == plan.plan_id
    assert payload.preset == plan.preset
    assert payload.product == plan.product
    assert payload.runtime_target_seconds == pytest.approx(8.0)
    assert payload.storyboard_deck == plan.storyboard_deck
    assert payload.status == FirstFrameReviewStatus.AWAITING_REVIEW
    assert payload.approval_gate.take_generation_blocked is True
    assert payload.approval_gate.motion_generation_blocked is True

    assert len(payload.shots) == plan.shot_count == 6
    for payload_shot, plan_shot in zip(payload.shots, plan.shots, strict=True):
        assert payload_shot.shot_id == plan_shot.shot_id
        assert payload_shot.sequence_order == plan_shot.sequence_order
        assert payload_shot.role == plan_shot.role
        assert payload_shot.intent == plan_shot.intent
        assert payload_shot.duration_target_seconds == plan_shot.duration_target_seconds
        assert payload_shot.storyboard_deck == plan.storyboard_deck
        assert payload_shot.storyboard_board == plan_shot.storyboard_board
        assert payload_shot.reference_requirements == plan_shot.reference_requirements
        assert payload_shot.prompt_inputs.storyboard_deck == plan.storyboard_deck
        assert payload_shot.prompt_inputs.storyboard_board == plan_shot.storyboard_board
        assert payload_shot.prompt_inputs.motion_beat_after_approval == plan_shot.motion_beat
        assert payload_shot.prompt_inputs.preserve_approved_board_text is True
        assert payload_shot.status == FirstFrameReviewStatus.AWAITING_REVIEW
        assert payload_shot.approval_gate.take_generation_blocked is True
        assert payload_shot.approval_gate.motion_generation_blocked is True
        assert "storyboard board still for review" in payload_shot.review_prompt_text
        assert "designed beat-deck board" in payload_shot.review_prompt_text
        assert "Sequence continuity locks:" in payload_shot.review_prompt_text
        assert "Readable on-frame board copy" in payload_shot.review_prompt_text
        assert "Sequence lock reference assets to preserve across every board" in (
            payload_shot.review_prompt_text
        )
        assert "Do not create video, animation, motion, or take variations yet." in (
            payload_shot.review_prompt_text
        )


def test_tall_owl_payload_preserves_real_product_brand_and_audience_locks() -> None:
    """Tall Owl benchmark payload carries concrete review inputs before motion."""
    payload = build_tall_owl_first_frame_review_payload()
    prompt_text = "\n".join(shot.review_prompt_text for shot in payload.shots)
    asset_ids = {asset.asset_id for asset in payload.reference_assets}
    asset_roles = {asset.role for asset in payload.reference_assets}

    assert payload.product.product_name == "Tall Owl Whipped Tallow Vanilla Orange"
    assert payload.brand
    assert payload.brand.brand_name == "Tall Owl"
    assert payload.aspect_ratio == "9:16"
    assert len(payload.reference_assets) == 4
    assert "tall_owl_vanilla_orange_packaging_front" in asset_ids
    assert "tall_owl_owl_mascot_hires" in asset_ids
    assert "product_packaging_lock" in asset_roles
    assert "mascot_logo_lock" in asset_roles

    assert "real Tall Owl owl logo mark" in prompt_text
    assert "do not invent a new owl" in prompt_text
    assert "Tall Owl Whipped Tallow Vanilla Orange jar packaging" in prompt_text
    assert "VANILLA ORANGE in white caps on an orange brushstroke" in prompt_text
    assert "Net Wt. 3.0 oz" in prompt_text
    assert "same locked jar/owl/logo subject set across the full sequence" in prompt_text
    assert "Sequence visual world" in prompt_text
    assert "Sequence layout system" in prompt_text
    assert "Sequence continuity locks:" in prompt_text
    assert "Recurring hero subject / product lock across every board" in prompt_text
    assert 'Readable on-frame board copy: "deep moisture for dry, sensitive, reactive skin"' in (
        prompt_text
    )
    assert 'Readable on-frame board copy: "simplifying a dry, reactive skincare routine"' in (
        prompt_text
    )
    assert 'Readable on-frame board copy: "Tall Owl"' in prompt_text
    assert "dry, sensitive, reactive skin" in prompt_text
    assert "not waxy, heavy, or greasy" in prompt_text
    assert "generic luxury skincare" in prompt_text
    assert "Take generation remains blocked" in prompt_text

    lower_prompt = prompt_text.lower()
    assert "google" not in lower_prompt
    assert "gemini" not in lower_prompt
    assert "luma" not in lower_prompt
    assert "kling" not in lower_prompt

    for shot in payload.shots:
        assert shot.reference_asset_ids == [asset.asset_id for asset in payload.reference_assets]
        assert shot.prompt_inputs.product_name == "Tall Owl Whipped Tallow Vanilla Orange"
        assert shot.prompt_inputs.reference_asset_ids == shot.reference_asset_ids
        assert shot.prompt_inputs.storyboard_board.on_frame_text
        assert shot.storyboard_board.on_frame_text
        assert shot.approval_gate.reason.startswith("Take generation is blocked")


def test_tall_owl_first_frame_review_cli_prints_json() -> None:
    """The CLI exposes the narrow Tall Owl review-payload dry run."""
    runner = CliRunner()

    result = runner.invoke(app, ["shorts", "tall-owl-first-frame-review"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == "first-frame-review.v1"
    assert data["source_plan_id"].startswith("shotplan_premium_product_macro_reveal_packshot")
    assert data["product"]["product_name"] == "Tall Owl Whipped Tallow Vanilla Orange"
    assert data["brand"]["brand_name"] == "Tall Owl"
    assert len(data["shots"]) == 6
    assert data["shots"][0]["approval_gate"]["take_generation_blocked"] is True
    assert data["shots"][0]["approval_gate"]["motion_generation_blocked"] is True
