"""Tests for typed ref-pack generation."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shorts_engine.contracts.ref_pack_v1 import REF_PACK_SCHEMA_VERSION, RefPackV1
from shorts_engine.services.ref_pack_generator import GeneratedReferenceImage, RefPackGenerator
from shorts_engine.shot_plans.benchmarks import (
    TALL_OWL_REFERENCE_ASSETS,
    build_tall_owl_first_frame_review_payload,
    compile_tall_owl_benchmark_shot_plan,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlH0i8AAAAASUVORK5CYII="
)


class FakeReferenceImageGenerator:
    """Deterministic fake used to prove ref-pack contract generation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "prompt": prompt,
                "candidate_index": candidate_index,
                "aspect_ratio": aspect_ratio,
                "model": model,
                "shot_id": shot.shot_id,
                "reference_asset_ids": [asset.asset_id for asset in reference_assets],
            }
        )
        return GeneratedReferenceImage(
            image_bytes=PNG_BYTES,
            mime_type="image/png",
            provider_params={
                "generator": "fake",
                "candidate_index": candidate_index,
            },
        )


@pytest.mark.asyncio
async def test_fixture_shot_plan_generates_valid_ref_pack_with_lineage(tmp_path: Path) -> None:
    """A valid compiled shot plan can emit a typed `ref_pack.v1` artifact."""
    plan = compile_tall_owl_benchmark_shot_plan()
    generator = FakeReferenceImageGenerator()
    created_at = datetime(2026, 4, 21, 18, 47, tzinfo=UTC)
    service = RefPackGenerator(
        image_generator=generator,
        output_root=tmp_path / "references",
        now_factory=lambda: created_at,
    )

    artifact = await service.generate(
        job_id="job-demo-001",
        shot_plan=plan,
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
    )
    round_tripped = RefPackV1.model_validate_json(artifact.model_dump_json())
    expected_asset_ids = [asset.asset_id for asset in TALL_OWL_REFERENCE_ASSETS]

    assert round_tripped == artifact
    assert artifact.schema_version == REF_PACK_SCHEMA_VERSION
    assert artifact.job_id == "job-demo-001"
    assert artifact.preset_id == plan.preset.preset_id
    assert artifact.source_shot_plan_id == plan.plan_id
    assert artifact.lineage.preset_id == plan.preset.preset_id
    assert artifact.lineage.preset_version == plan.preset.version
    assert artifact.lineage.source_plan_id == plan.plan_id
    assert artifact.lineage.aspect_ratio == "9:16"
    assert artifact.lineage.reference_asset_ids == expected_asset_ids
    assert artifact.lineage.source_review_payload_id.startswith("firstframe_")
    assert len(artifact.shots) == plan.shot_count == 3
    assert len(generator.calls) == 9

    for shot_group, plan_shot in zip(artifact.shots, plan.shots, strict=True):
        assert shot_group.shot_id == plan_shot.shot_id
        assert len(shot_group.reference_candidates) == 3

        for candidate_index, candidate in enumerate(shot_group.reference_candidates, start=1):
            asset_path = Path(candidate.asset_path)
            assert asset_path.exists()
            assert "job-demo-001" in candidate.asset_path
            assert plan_shot.shot_id in candidate.asset_path
            assert candidate.ref_id == f"{plan_shot.shot_id}_ref_{candidate_index:02d}"
            assert candidate.model == "nano-banana-pro-preview"
            assert candidate.created_at == created_at
            assert candidate.params.aspect_ratio == "9:16"
            assert candidate.params.candidate_index == candidate_index
            assert candidate.params.first_frame_prompt_id == f"{plan_shot.shot_id}_first_frame_prompt"
            assert candidate.params.reference_asset_ids == expected_asset_ids
            assert candidate.params.provider_params["generator"] == "fake"
            assert "before any motion or take generation" in candidate.params.prompt
            assert candidate.prompt_summary.startswith(f"Shot {plan_shot.sequence_order}")


@pytest.mark.asyncio
async def test_ref_pack_shape_is_stable_and_machine_readable(tmp_path: Path) -> None:
    """The serialized contract stays small, explicit, and UI-agnostic."""
    plan = compile_tall_owl_benchmark_shot_plan()
    service = RefPackGenerator(
        image_generator=FakeReferenceImageGenerator(),
        output_root=tmp_path / "references",
        now_factory=lambda: datetime(2026, 4, 21, 18, 47, tzinfo=UTC),
    )

    artifact = await service.generate(
        job_id="job-shape-001",
        shot_plan=plan,
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
    )
    data = artifact.model_dump(mode="json")

    assert json.loads(artifact.model_dump_json()) == data
    assert set(data) == {
        "ref_pack_id",
        "schema_version",
        "job_id",
        "preset_id",
        "source_shot_plan_id",
        "lineage",
        "shots",
    }
    assert set(data["lineage"]) == {
        "preset_id",
        "preset_version",
        "source_plan_id",
        "source_review_payload_id",
        "aspect_ratio",
        "reference_asset_ids",
        "review_guidance",
    }
    assert set(data["shots"][0]) == {"shot_id", "reference_candidates"}
    assert set(data["shots"][0]["reference_candidates"][0]) == {
        "ref_id",
        "asset_path",
        "prompt_summary",
        "model",
        "params",
        "created_at",
        "status",
    }


@pytest.mark.asyncio
async def test_ref_pack_can_be_generated_from_existing_review_payload(tmp_path: Path) -> None:
    """The generator can consume a prebuilt first-frame review payload directly."""
    review_payload = build_tall_owl_first_frame_review_payload()
    generator = FakeReferenceImageGenerator()
    created_at = datetime(2026, 4, 21, 18, 47, tzinfo=UTC)
    service = RefPackGenerator(
        image_generator=generator,
        output_root=tmp_path / "references",
        now_factory=lambda: created_at,
    )

    artifact = await service.generate(
        job_id="job-review-payload-001",
        review_payload=review_payload.model_dump(mode="json"),
    )

    assert artifact.source_shot_plan_id == review_payload.source_plan_id
    assert artifact.preset_id == review_payload.preset.preset_id
    assert artifact.lineage.source_review_payload_id == review_payload.payload_id
    assert artifact.lineage.review_guidance == review_payload.review_guidance
    assert artifact.lineage.reference_asset_ids == [
        asset.asset_id for asset in review_payload.reference_assets
    ]
    assert len(artifact.shots) == len(review_payload.shots)
    assert len(generator.calls) == 9
