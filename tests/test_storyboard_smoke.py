"""Tests for the storyboard-first smoke runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shorts_engine.cli import app
from shorts_engine.services.storyboard_smoke import run_storyboard_first_smoke
from shorts_engine.shot_plans.benchmarks import TALL_OWL_BENCHMARK_ID


@pytest.mark.asyncio
async def test_storyboard_first_smoke_writes_full_artifact_bundle(tmp_path: Path) -> None:
    """The smoke runner composes plan, review, refs, approvals, and takes in one run."""
    result = await run_storyboard_first_smoke(
        output_root=tmp_path,
        job_id="storyboard-smoke-test",
        candidates_per_shot=2,
        take_count=1,
    )

    assert result.benchmark_id == TALL_OWL_BENCHMARK_ID
    assert result.job_id == "storyboard-smoke-test"
    assert result.selection_strategy == "first_candidate_per_shot"
    assert len(result.approved_ref_ids_by_shot) == 6
    assert len(result.reference_candidate_asset_paths) == 12
    assert len(result.motion_take_asset_paths) == 6

    compiled_plan = json.loads(Path(result.artifacts.compiled_shot_plan_path).read_text())
    review_payload = json.loads(Path(result.artifacts.first_frame_review_payload_path).read_text())
    ref_pack = json.loads(Path(result.artifacts.ref_pack_path).read_text())
    approvals = json.loads(Path(result.artifacts.approvals_path).read_text())
    approved_plan = json.loads(Path(result.artifacts.approved_shot_plan_path).read_text())
    summary = json.loads(Path(result.artifacts.summary_path).read_text())

    assert compiled_plan["schema_version"] == "shot-plan.v1"
    assert review_payload["schema_version"] == "first-frame-review.v1"
    assert ref_pack["schema_version"] == "ref_pack.v1"
    assert approvals["selection_strategy"] == "first_candidate_per_shot"
    assert approved_plan["status"] == "ready_for_takes"
    assert summary["plan_id"] == compiled_plan["plan_id"]
    assert summary["review_payload_id"] == review_payload["payload_id"]
    assert summary["ref_pack_id"] == ref_pack["ref_pack_id"]
    assert len(compiled_plan["storyboard_deck"]["continuity_locks"]) == 4
    assert "Recurring hero subject / product lock across every board" in (
        review_payload["shots"][0]["review_prompt_text"]
    )

    assert len(ref_pack["shots"]) == 6
    for shot_group in ref_pack["shots"]:
        assert len(shot_group["reference_candidates"]) == 2
        for candidate in shot_group["reference_candidates"]:
            candidate_path = Path(candidate["asset_path"])
            assert candidate["status"] == "generated"
            assert candidate["model"] == "storyboard-smoke-reference-v1"
            assert candidate_path.exists()
            assert candidate_path.suffix == ".png"

    for batch_path in result.artifacts.shot_take_batch_paths:
        batch = json.loads(Path(batch_path).read_text())
        assert batch["schema_version"] == "shot_take.v1"
        assert len(batch["takes"]) == 1
        take = batch["takes"][0]
        assert take["status"] == "generated"
        assert take["model"] == "storyboard-smoke-motion-v1"
        assert take["asset_path"].endswith(".gif")
        assert Path(take["asset_path"]).exists()


def test_storyboard_first_smoke_cli_runs_end_to_end(tmp_path: Path) -> None:
    """The CLI exposes the narrow smoke command with inspectable outputs."""
    runner = CliRunner()
    output_root = tmp_path / "storyboard-smoke-cli"

    result = runner.invoke(
        app,
        [
            "shorts",
            "storyboard-first-smoke",
            "--output-root",
            str(output_root),
            "--job-id",
            "storyboard-smoke-cli",
            "--candidates-per-shot",
            "2",
            "--take-count",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Storyboard-first smoke completed" in result.output
    summary_path = output_root / "storyboard-smoke-cli" / "storyboard_first_smoke_summary.json"
    assert summary_path.exists()

    data = json.loads(summary_path.read_text())
    assert data["benchmark_id"] == TALL_OWL_BENCHMARK_ID
    assert len(data["reference_candidate_asset_paths"]) == 12
    assert len(data["motion_take_asset_paths"]) == 6
