"""Tests for typed shot-take generation."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shorts_engine.adapters.video_gen.base import VideoGenProvider, VideoGenRequest, VideoGenResult
from shorts_engine.contracts.shot_take_v1 import (
    SHOT_TAKE_SCHEMA_VERSION,
    ShotTakeBatchV1,
    ShotTakeStatus,
)
from shorts_engine.jobs import video_pipeline
from shorts_engine.services.ref_pack_generator import GeneratedReferenceImage, RefPackGenerator
from shorts_engine.services.shot_generation_runner import ShotGenerationRunner
from shorts_engine.shot_plans import (
    FLAGSHIP_PRESET_ID,
    FLAGSHIP_PRESET_VERSION,
    TALL_OWL_REFERENCE_ASSETS,
    ApprovedStoryboardBoard,
    apply_ref_pack_approvals,
)
from shorts_engine.shot_plans.benchmarks import compile_tall_owl_benchmark_shot_plan
from shorts_engine.shot_plans.contracts import TakeRequestStatus

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlH0i8AAAAASUVORK5CYII="
)


class FakeVideoGenProvider(VideoGenProvider):
    """Deterministic fake that captures the existing generation seam."""

    def __init__(
        self,
        results: list[VideoGenResult],
        *,
        model: str = "fake-video-model",
        supports_reference_images: bool = True,
    ) -> None:
        self._results = results
        self.model = model
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

    async def check_status(self, job_id: str) -> dict[str, str]:
        return {"job_id": job_id, "status": "completed"}


class FakeReferenceImageGenerator:
    """Deterministic fake used to build a real ref-pack in the runner tests."""

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


def _approved_reference_paths(tmp_path: Path) -> list[str]:
    reference_dir = tmp_path / "approved_refs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(1, 3):
        path = reference_dir / f"ref_{index:02d}.png"
        path.write_bytes(PNG_BYTES)
        paths.append(str(path))
    return paths


def _product_input() -> dict[str, str | list[str]]:
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


def _brand_input() -> dict[str, str]:
    return {
        "brand_name": "Aurum",
        "brand_voice": "premium, restrained, sensory",
        "visual_style": "cinematic commercial realism",
        "environment": "warm marble bathroom counter with soft window light",
    }


@pytest.mark.asyncio
async def test_take_request_emits_multiple_typed_take_records_with_lineage(tmp_path: Path) -> None:
    """One ready request yields a stable typed batch of take artifacts."""
    plan = compile_tall_owl_benchmark_shot_plan()
    created_at = datetime(2026, 4, 21, 19, 1, tzinfo=UTC)
    ref_pack = await RefPackGenerator(
        image_generator=FakeReferenceImageGenerator(),
        output_root=tmp_path / "references",
        now_factory=lambda: created_at,
    ).generate(
        job_id="job-demo-001",
        shot_plan=plan,
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
        candidates_per_shot=2,
    )
    approved_plan = apply_ref_pack_approvals(
        plan,
        ref_pack,
        {plan.shots[0].shot_id: ref_pack.shots[0].reference_candidates[0].ref_id},
    )
    take_request = approved_plan.shots[0].take_request
    reference_paths = [
        candidate.asset_path for candidate in ref_pack.shots[0].reference_candidates
    ]
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=f"FAKE_MP4_TAKE_{take_index}".encode(),
                metadata={
                    "generation_id": f"gen-{take_index:02d}",
                    "model": "fake-video-model",
                    "cost_estimate": 0.25,
                },
            )
            for take_index in range(1, 4)
        ]
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: created_at,
    )

    artifact = await runner.generate(
        job_id="job-demo-001",
        take_request=take_request.model_dump(mode="json"),
        reference_asset_paths=reference_paths,
    )
    round_tripped = ShotTakeBatchV1.model_validate_json(artifact.model_dump_json())
    data = artifact.model_dump(mode="json")

    assert round_tripped == artifact
    assert artifact.schema_version == SHOT_TAKE_SCHEMA_VERSION
    assert artifact.job_id == "job-demo-001"
    assert artifact.shot_id == take_request.shot_id
    assert artifact.take_request_id == take_request.take_request_id
    assert len(artifact.takes) == 3
    assert len(provider.requests) == 3
    assert json.loads(artifact.model_dump_json()) == data
    assert set(data) == {
        "take_batch_id",
        "schema_version",
        "job_id",
        "shot_id",
        "take_request_id",
        "takes",
    }
    assert set(data["takes"][0]) == {
        "take_id",
        "job_id",
        "shot_id",
        "take_request_id",
        "asset_path",
        "model",
        "params",
        "seed",
        "cost_estimate",
        "created_at",
        "status",
        "error_message",
        "lineage",
        "provider_metadata",
    }

    seeds = [take.seed for take in artifact.takes]
    assert all(seed is not None for seed in seeds)
    assert len(set(seeds)) == 3

    for take_index, take in enumerate(artifact.takes, start=1):
        asset_path = Path(str(take.asset_path))
        assert asset_path.exists()
        assert take.take_id == f"{take_request.take_request_id}_take_{take_index:02d}"
        assert "job-demo-001" in str(asset_path)
        assert take_request.shot_id in str(asset_path)
        assert take.model == "fake-video-model"
        assert take.cost_estimate == pytest.approx(0.25)
        assert take.created_at == created_at
        assert take.status == ShotTakeStatus.GENERATED
        assert take.error_message is None
        assert take.params.take_index == take_index
        assert take.params.duration_seconds == take_request.duration_target_seconds
        assert take.params.reference_asset_paths == reference_paths
        assert take.params.provider_params["seed"] == take.seed
        assert take.lineage.preset_id == take_request.preset_id
        assert take.lineage.preset_version == take_request.preset_version
        assert take.lineage.sequence_order == take_request.sequence_order
        assert take.lineage.source_plan_id == plan.plan_id
        assert (
            take.lineage.approved_board_ref_id
            == ref_pack.shots[0].reference_candidates[0].ref_id
        )
        assert take.lineage.approved_board_asset_path == reference_paths[0]
        assert take.lineage.approved_board_source_ref_pack_id == ref_pack.ref_pack_id
        assert (
            take.lineage.approved_board_source_review_payload_id
            == ref_pack.lineage.source_review_payload_id
        )
        assert take.lineage.reference_asset_paths == reference_paths
        assert take.lineage.provider_job_id == f"gen-{take_index:02d}"
        assert take.lineage.request_metadata["requires_review"] is True

    assert isinstance(provider.requests[0], VideoGenRequest)
    assert provider.requests[0].reference_images is not None
    assert len(provider.requests[0].reference_images) == 2
    assert provider.requests[0].negative_prompt is not None
    assert "approved storyboard board / first-frame direction" in provider.requests[0].prompt
    assert "Storyboard visual world" in provider.requests[0].prompt
    assert "Sequence continuity locks" in provider.requests[0].prompt
    assert "Recurring hero subject / product lock across every board" in provider.requests[0].prompt
    assert "Preserve this approved on-frame board copy exactly" in provider.requests[0].prompt
    assert "changed board copy" in provider.requests[0].negative_prompt


@pytest.mark.asyncio
async def test_pipeline_take_request_handoff_preserves_approved_board_lineage(
    tmp_path: Path,
) -> None:
    """The normal shot-plan caller path keeps approved-board lineage intact into take artifacts."""
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
        job_id="job-pipeline-handoff-001",
        shot_plan=compiled,
        candidates_per_shot=2,
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

    take_request = video_plan.scenes[0].metadata["shot_plan"]["take_request"]
    reference_paths = [
        candidate.asset_path for candidate in ref_pack.shots[0].reference_candidates
    ]
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"PIPELINE_HANDOFF_TAKE",
                metadata={
                    "generation_id": "gen-pipeline-handoff-01",
                    "model": "fake-video-model",
                },
            )
        ]
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: created_at,
    )

    artifact = await runner.generate(
        job_id="job-pipeline-handoff-001",
        take_request=take_request,
        reference_asset_paths=reference_paths,
        take_count=1,
    )

    take = artifact.takes[0]
    assert take.status == ShotTakeStatus.GENERATED
    assert take.lineage.source_plan_id == compiled.plan_id
    assert take.lineage.approved_board_ref_id == approved_candidate.ref_id
    assert take.lineage.approved_board_asset_path == approved_candidate.asset_path
    assert take.lineage.approved_board_source_ref_pack_id == ref_pack.ref_pack_id
    assert (
        take.lineage.approved_board_source_review_payload_id
        == ref_pack.lineage.source_review_payload_id
    )
    assert take.params.reference_asset_paths[0] == approved_candidate.asset_path
    assert take.lineage.reference_asset_paths[0] == approved_candidate.asset_path


@pytest.mark.asyncio
async def test_binary_provider_options_are_sanitized_in_artifact_but_preserved_for_provider(
    tmp_path: Path,
) -> None:
    """Binary provider options should reach the provider but stay JSON-safe in artifacts."""
    plan = compile_tall_owl_benchmark_shot_plan()
    reference_paths = _approved_reference_paths(tmp_path)
    take_request = plan.shots[0].take_request.model_copy(
        update={
            "status": TakeRequestStatus.READY,
            "approved_board": ApprovedStoryboardBoard(
                ref_id=f"{plan.shots[0].shot_id}_ref_01",
                asset_path=reference_paths[0],
            ),
        }
    )
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"BINARY_OPTION_TAKE",
                metadata={"generation_id": "gen-binary-01", "model": "fake-video-model"},
            )
        ]
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: datetime(2026, 4, 21, 19, 1, tzinfo=UTC),
    )

    artifact = await runner.generate(
        job_id="job-binary-provider-option-001",
        take_request=take_request,
        reference_asset_paths=reference_paths,
        take_count=1,
        provider_params={"end_reference_image": PNG_BYTES},
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].options is not None
    assert provider.requests[0].options["end_reference_image"] == PNG_BYTES
    assert artifact.takes[0].params.provider_params["end_reference_image"] == (
        f"<binary:{len(PNG_BYTES)} bytes>"
    )

    serialized = json.loads(artifact.model_dump_json())
    assert serialized["takes"][0]["params"]["provider_params"]["end_reference_image"] == (
        f"<binary:{len(PNG_BYTES)} bytes>"
    )


@pytest.mark.asyncio
async def test_failed_takes_are_kept_in_the_batch_instead_of_dropped(tmp_path: Path) -> None:
    """Failed take attempts stay visible in the artifact output."""
    plan = compile_tall_owl_benchmark_shot_plan()
    reference_paths = _approved_reference_paths(tmp_path)
    take_request = plan.shots[1].take_request.model_copy(
        update={
            "status": TakeRequestStatus.READY,
            "approved_board": ApprovedStoryboardBoard(
                ref_id=f"{plan.shots[1].shot_id}_ref_01",
                asset_path=reference_paths[0],
            ),
        }
    )
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"FIRST_TAKE",
                metadata={"generation_id": "gen-01", "model": "fake-video-model"},
            ),
            VideoGenResult(
                success=False,
                error_message="provider refused prompt",
                metadata={"generation_id": "gen-02", "model": "fake-video-model"},
            ),
            VideoGenResult(
                success=True,
                video_data=b"THIRD_TAKE",
                metadata={"generation_id": "gen-03", "model": "fake-video-model"},
            ),
        ]
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: datetime(2026, 4, 21, 19, 1, tzinfo=UTC),
    )

    artifact = await runner.generate(
        job_id="job-failure-001",
        take_request=take_request,
        reference_asset_paths=reference_paths,
    )

    assert len(artifact.takes) == 3
    assert [take.status for take in artifact.takes] == [
        ShotTakeStatus.GENERATED,
        ShotTakeStatus.FAILED,
        ShotTakeStatus.GENERATED,
    ]
    failed_take = artifact.takes[1]
    assert failed_take.take_id == f"{take_request.take_request_id}_take_02"
    assert failed_take.job_id == "job-failure-001"
    assert failed_take.shot_id == take_request.shot_id
    assert failed_take.take_request_id == take_request.take_request_id
    assert failed_take.asset_path is None
    assert failed_take.error_message == "provider refused prompt"
    assert failed_take.lineage.provider_job_id == "gen-02"
    assert failed_take.lineage.preset_id == take_request.preset_id
    assert failed_take.params.take_index == 2
    assert len(provider.requests) == 3


@pytest.mark.asyncio
async def test_take_request_must_be_ready_before_generation_runs(tmp_path: Path) -> None:
    """Only READY requests may produce take attempts."""
    plan = compile_tall_owl_benchmark_shot_plan()
    reference_paths = _approved_reference_paths(tmp_path)
    take_request = plan.shots[0].take_request.model_copy(
        update={
            "status": TakeRequestStatus.REQUESTED,
            "approved_board": ApprovedStoryboardBoard(
                ref_id=f"{plan.shots[0].shot_id}_ref_01",
                asset_path=reference_paths[0],
            ),
        }
    )
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"UNUSED",
                metadata={"generation_id": "gen-01", "model": "fake-video-model"},
            )
        ]
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: datetime(2026, 4, 21, 19, 1, tzinfo=UTC),
    )

    artifact = await runner.generate(
        job_id="job-not-ready-001",
        take_request=take_request,
        reference_asset_paths=reference_paths,
        take_count=1,
    )

    assert len(provider.requests) == 0
    assert len(artifact.takes) == 1
    assert artifact.takes[0].status == ShotTakeStatus.FAILED
    assert artifact.takes[0].error_message == (
        "Take request must be READY before generation (got requested)"
    )


@pytest.mark.asyncio
async def test_take_request_requires_an_approved_storyboard_board(tmp_path: Path) -> None:
    """Ready requests still stay blocked until a specific approved board exists."""
    plan = compile_tall_owl_benchmark_shot_plan()
    take_request = plan.shots[0].take_request.model_copy(update={"status": TakeRequestStatus.READY})
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"UNUSED",
                metadata={"generation_id": "gen-01", "model": "fake-video-model"},
            )
        ]
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: datetime(2026, 4, 21, 19, 1, tzinfo=UTC),
    )

    artifact = await runner.generate(
        job_id="job-missing-board-001",
        take_request=take_request,
        reference_asset_paths=_approved_reference_paths(tmp_path),
        take_count=1,
    )

    assert len(provider.requests) == 0
    assert len(artifact.takes) == 1
    assert artifact.takes[0].status == ShotTakeStatus.FAILED
    assert artifact.takes[0].error_message == "Approved storyboard board is required before take generation"


@pytest.mark.asyncio
async def test_take_request_requires_provider_reference_image_support(tmp_path: Path) -> None:
    """Shot-plan take generation must fail closed when the provider cannot accept references."""
    plan = compile_tall_owl_benchmark_shot_plan()
    reference_paths = _approved_reference_paths(tmp_path)
    take_request = plan.shots[0].take_request.model_copy(
        update={
            "status": TakeRequestStatus.READY,
            "approved_board": ApprovedStoryboardBoard(
                ref_id=f"{plan.shots[0].shot_id}_ref_01",
                asset_path=reference_paths[0],
            ),
        }
    )
    provider = FakeVideoGenProvider(
        [
            VideoGenResult(
                success=True,
                video_data=b"UNUSED",
                metadata={"generation_id": "gen-01", "model": "fake-video-model"},
            )
        ],
        supports_reference_images=False,
    )
    runner = ShotGenerationRunner(
        video_provider=provider,
        output_root=tmp_path / "shot_takes",
        now_factory=lambda: datetime(2026, 4, 21, 19, 1, tzinfo=UTC),
    )

    artifact = await runner.generate(
        job_id="job-provider-capability-001",
        take_request=take_request,
        reference_asset_paths=reference_paths,
        take_count=1,
    )

    assert len(provider.requests) == 0
    assert len(artifact.takes) == 1
    assert artifact.takes[0].status == ShotTakeStatus.FAILED
    assert artifact.takes[0].error_message == (
        "Configured video provider does not support approved-board reference inputs "
        "required for shot-plan generation"
    )
