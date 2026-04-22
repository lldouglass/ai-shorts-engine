"""Tests for preset-driven shot-plan compilation."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from shorts_engine.cli import app
from shorts_engine.services.planner import PlannerService
from shorts_engine.shot_plans import (
    FLAGSHIP_PRESET_ID,
    FLAGSHIP_PRESET_VERSION,
    PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT,
    CompiledShotPlan,
    PresetSpec,
    ShotStatus,
    TakeRequestStatus,
    compile_shot_plan,
)

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "shot_plans"
    / "premium_product_macro_reveal_packshot_v1.json"
)


def _product_input() -> dict[str, Any]:
    return {
        "product_name": "Aurum Glow Serum",
        "product_category": "premium skincare",
        "concept": "Premium serum product short",
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


def test_flagship_preset_fixture_validates() -> None:
    """The checked-in flagship preset fixture validates against the contract."""
    preset = PresetSpec.model_validate_json(FIXTURE_PATH.read_text())

    assert preset.model_dump(mode="json") == PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT.model_dump(
        mode="json"
    )
    assert preset.preset_id == FLAGSHIP_PRESET_ID
    assert preset.version == FLAGSHIP_PRESET_VERSION
    assert [shot.role for shot in preset.shot_templates] == [
        "macro_hook",
        "reveal_demo",
        "packshot_payoff",
    ]


def test_compile_output_is_deterministic() -> None:
    """The same preset and inputs emit byte-for-byte equivalent JSON."""
    first = compile_shot_plan(
        FLAGSHIP_PRESET_ID,
        FLAGSHIP_PRESET_VERSION,
        product=_product_input(),
        brand=_brand_input(),
    )
    second = compile_shot_plan(
        FLAGSHIP_PRESET_ID,
        FLAGSHIP_PRESET_VERSION,
        product=_product_input(),
        brand=_brand_input(),
    )

    assert first.plan_id == second.plan_id
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.model_dump_json() == second.model_dump_json()


def test_compiler_emits_clean_three_shot_plan() -> None:
    """The flagship compiler emits a downstream-ready three-shot package."""
    plan = compile_shot_plan(
        FLAGSHIP_PRESET_ID,
        FLAGSHIP_PRESET_VERSION,
        product=_product_input(),
        brand=_brand_input(),
    )
    round_tripped = CompiledShotPlan.model_validate_json(plan.model_dump_json())

    assert round_tripped == plan
    assert plan.shot_count == 3
    assert plan.runtime_target_seconds == pytest.approx(8.0)
    assert sum(shot.duration_target_seconds for shot in plan.shots) == pytest.approx(8.0)
    assert [shot.role for shot in plan.shots] == [
        "macro_hook",
        "reveal_demo",
        "packshot_payoff",
    ]

    for expected_order, shot in enumerate(plan.shots, start=1):
        assert shot.sequence_order == expected_order
        assert shot.shot_id
        assert shot.intent
        assert "Aurum Glow Serum" in shot.subject
        assert "{" not in shot.subject
        assert "{" not in shot.environment
        assert "{" not in shot.motion_beat
        assert shot.camera_language
        assert shot.duration_target_seconds > 0
        assert shot.reference_requirements
        assert shot.take_request
        assert shot.status == ShotStatus.NEEDS_REFERENCES


def test_take_request_metadata_supports_refs_takes_and_review() -> None:
    """Each shot carries enough metadata for reference approval and take generation."""
    plan = compile_shot_plan(
        FLAGSHIP_PRESET_ID,
        FLAGSHIP_PRESET_VERSION,
        product=_product_input(),
        brand=_brand_input(),
    )

    for shot in plan.shots:
        take_request = shot.take_request
        assert take_request.take_request_id == f"{shot.shot_id}_take_request"
        assert take_request.shot_id == shot.shot_id
        assert take_request.preset_id == FLAGSHIP_PRESET_ID
        assert take_request.preset_version == FLAGSHIP_PRESET_VERSION
        assert take_request.sequence_order == shot.sequence_order
        assert take_request.intent == shot.intent
        assert take_request.role == shot.role
        assert take_request.subject == shot.subject
        assert take_request.environment == shot.environment
        assert take_request.motion_beat == shot.motion_beat
        assert take_request.camera_language == shot.camera_language
        assert take_request.duration_target_seconds == shot.duration_target_seconds
        assert take_request.reference_requirements == shot.reference_requirements
        assert take_request.status == TakeRequestStatus.BLOCKED_ON_REFERENCES

        assert take_request.generation_defaults.target_take_count == 3
        assert take_request.generation_defaults.seed_policy == "deterministic_per_shot_take"
        assert take_request.generation_defaults.requires_approved_reference is True
        assert take_request.generation_defaults.avoid_visible_text is True
        assert take_request.generation_defaults.variation_axes
        assert take_request.variation_hints
        assert take_request.metadata["requires_review"] is True
        assert take_request.metadata["source_plan_id"] == plan.plan_id

        for requirement in take_request.reference_requirements:
            assert requirement.approval_required is True
            assert requirement.required_before == "take_generation"
            assert requirement.count == 3
            assert "{" not in requirement.description


def test_planner_service_compiles_flagship_shot_plan() -> None:
    """PlannerService exposes the deterministic preset compiler as a public seam."""
    plan = PlannerService.compile_shot_plan(product=_product_input(), brand=_brand_input())

    assert isinstance(plan, CompiledShotPlan)
    assert plan.preset.preset_id == FLAGSHIP_PRESET_ID
    assert plan.preset.version == FLAGSHIP_PRESET_VERSION
    assert plan.product.product_name == "Aurum Glow Serum"
    assert plan.brand
    assert plan.brand.brand_name == "Aurum"
    assert plan.shot_count == 3


def test_shorts_compile_shot_plan_cli_prints_json() -> None:
    """The shorts CLI can compile a preset shot plan to stdout."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "shorts",
            "compile-shot-plan",
            "--product-name",
            "Aurum Glow Serum",
            "--key-benefit",
            "a clean glass-skin glow in one drop",
            "--brand-name",
            "Aurum",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["preset"]["preset_id"] == FLAGSHIP_PRESET_ID
    assert data["product"]["product_name"] == "Aurum Glow Serum"
    assert data["brand"]["brand_name"] == "Aurum"
    assert len(data["shots"]) == 3


def test_shorts_compile_shot_plan_cli_writes_json(tmp_path: Path) -> None:
    """The shorts CLI can write compiled shot-plan JSON to a file."""
    runner = CliRunner()
    output_path = tmp_path / "shot-plan.json"

    result = runner.invoke(
        app,
        [
            "shorts",
            "compile-shot-plan",
            "--product-name",
            "Aurum Glow Serum",
            "--key-benefit",
            "a clean glass-skin glow in one drop",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote shot plan JSON" in result.output
    data = json.loads(output_path.read_text())
    assert data["preset"]["preset_id"] == FLAGSHIP_PRESET_ID
    assert data["product"]["key_benefit"] == "a clean glass-skin glow in one drop"
    assert len(data["shots"]) == 3
