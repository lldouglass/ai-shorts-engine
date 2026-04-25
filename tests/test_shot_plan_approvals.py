"""Tests for turning real ref-pack approvals into ready take requests."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from shorts_engine.services.ref_pack_generator import GeneratedReferenceImage, RefPackGenerator
from shorts_engine.shot_plans import TALL_OWL_REFERENCE_ASSETS, apply_ref_pack_approvals
from shorts_engine.shot_plans.benchmarks import compile_tall_owl_benchmark_shot_plan
from shorts_engine.shot_plans.contracts import ShotStatus, TakeRequestStatus

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


@pytest.mark.asyncio
async def test_ref_pack_approvals_make_take_requests_ready_with_real_lineage(
    tmp_path,
) -> None:
    """Real ref-pack selections populate approved_board on the compiled requests."""
    plan = compile_tall_owl_benchmark_shot_plan()
    ref_pack = await RefPackGenerator(
        image_generator=FakeReferenceImageGenerator(),
        output_root=tmp_path / "references",
        now_factory=lambda: datetime(2026, 4, 23, 20, 15, tzinfo=UTC),
    ).generate(
        job_id="job-demo-approvals-001",
        shot_plan=plan,
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
        candidates_per_shot=2,
    )
    approved_plan = apply_ref_pack_approvals(
        plan,
        ref_pack,
        {
            shot.shot_id: shot.reference_candidates[0].ref_id
            for shot in ref_pack.shots
        },
    )

    assert approved_plan.status == ShotStatus.READY_FOR_TAKES
    assert len(approved_plan.shots) == 6
    assert all(shot.status == ShotStatus.READY_FOR_TAKES for shot in approved_plan.shots)

    for shot, ref_group in zip(approved_plan.shots, ref_pack.shots, strict=True):
        approved_board = shot.take_request.approved_board

        assert shot.take_request.status == TakeRequestStatus.READY
        assert approved_board is not None
        assert approved_board.ref_id == ref_group.reference_candidates[0].ref_id
        assert approved_board.asset_path == ref_group.reference_candidates[0].asset_path
        assert approved_board.source_ref_pack_id == ref_pack.ref_pack_id
        assert approved_board.source_review_payload_id == ref_pack.lineage.source_review_payload_id
        assert (
            approved_board.first_frame_prompt_id
            == ref_group.reference_candidates[0].params.first_frame_prompt_id
        )


def test_ref_pack_approvals_reject_unknown_ref_ids() -> None:
    """Selections must point at a real candidate within the matching shot group."""
    plan = compile_tall_owl_benchmark_shot_plan()

    with pytest.raises(ValueError, match='Unknown approved ref_id "missing-ref-id"'):
        apply_ref_pack_approvals(
            plan.model_dump(mode="json"),
            {
                "ref_pack_id": "refpack:test:001",
                "schema_version": "ref_pack.v1",
                "job_id": "job:test:001",
                "preset_id": plan.preset.preset_id,
                "source_shot_plan_id": plan.plan_id,
                "lineage": {
                    "preset_id": plan.preset.preset_id,
                    "preset_version": plan.preset.version,
                    "source_plan_id": plan.plan_id,
                    "source_review_payload_id": "firstframe:test:001",
                    "aspect_ratio": "9:16",
                    "reference_asset_ids": [],
                    "review_guidance": [],
                },
                "shots": [
                    {
                        "shot_id": plan.shots[0].shot_id,
                        "reference_candidates": [
                            {
                                "ref_id": "real-ref-id",
                                "asset_path": "tmp/reference.png",
                                "prompt_summary": "candidate",
                                "model": "fake-model",
                                "params": {
                                    "aspect_ratio": "9:16",
                                    "candidate_index": 1,
                                    "prompt": "prompt",
                                    "first_frame_prompt_id": "prompt:test:001",
                                    "reference_asset_ids": [],
                                    "provider_params": {},
                                },
                                "created_at": "2026-04-23T20:15:00.000Z",
                                "status": "generated",
                            }
                        ],
                    }
                ],
            },
            {plan.shots[0].shot_id: "missing-ref-id"},
        )
