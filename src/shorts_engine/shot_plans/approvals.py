"""Resolve approved storyboard-board selections from real ref-pack artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shorts_engine.contracts.ref_pack_v1 import RefPackV1, ShotReferenceGroup
from shorts_engine.shot_plans.contracts import (
    ApprovedStoryboardBoard,
    CompiledShotPlan,
    ShotSpec,
    ShotStatus,
    ShotTakeRequest,
    TakeRequestStatus,
)


def apply_ref_pack_approvals(
    shot_plan: CompiledShotPlan | Mapping[str, Any],
    ref_pack: RefPackV1 | Mapping[str, Any],
    approved_ref_ids_by_shot: Mapping[str, str],
) -> CompiledShotPlan:
    """Attach approved ref-pack selections to the compiled plan take requests."""
    plan = (
        shot_plan
        if isinstance(shot_plan, CompiledShotPlan)
        else CompiledShotPlan.model_validate(shot_plan)
    )
    pack = ref_pack if isinstance(ref_pack, RefPackV1) else RefPackV1.model_validate(ref_pack)

    if pack.source_shot_plan_id != plan.plan_id:
        raise ValueError(
            "Ref-pack source_shot_plan_id must match the compiled shot plan plan_id"
        )

    if pack.preset_id != plan.preset.preset_id:
        raise ValueError("Ref-pack preset_id must match the compiled shot plan preset_id")

    if not approved_ref_ids_by_shot:
        return plan

    plan_shot_ids = {shot.shot_id for shot in plan.shots}
    unknown_shot_ids = sorted(set(approved_ref_ids_by_shot) - plan_shot_ids)
    if unknown_shot_ids:
        raise ValueError(
            "Approved ref-pack selections contain unknown shot ids: "
            + ", ".join(unknown_shot_ids)
        )

    ref_pack_groups = {shot.shot_id: shot for shot in pack.shots}
    missing_ref_pack_shots = sorted(set(approved_ref_ids_by_shot) - set(ref_pack_groups))
    if missing_ref_pack_shots:
        raise ValueError(
            "Ref-pack is missing approved-selection shot ids: "
            + ", ".join(missing_ref_pack_shots)
        )

    updated_shots = [
        _apply_approved_ref_to_shot(
            shot,
            ref_group=ref_pack_groups[shot.shot_id],
            ref_pack=pack,
            approved_ref_id=approved_ref_ids_by_shot.get(shot.shot_id),
        )
        if shot.shot_id in approved_ref_ids_by_shot
        else shot
        for shot in plan.shots
    ]

    all_required_shots_ready = all(
        not shot.take_request.generation_defaults.requires_approved_reference
        or shot.take_request.approved_board is not None
        for shot in updated_shots
    )

    return plan.model_copy(
        update={
            "shots": updated_shots,
            "status": (
                ShotStatus.READY_FOR_TAKES
                if all_required_shots_ready
                else ShotStatus.NEEDS_REFERENCES
            ),
        }
    )


def _apply_approved_ref_to_shot(
    shot: ShotSpec,
    *,
    ref_group: ShotReferenceGroup,
    ref_pack: RefPackV1,
    approved_ref_id: str | None,
) -> ShotSpec:
    if approved_ref_id is None:
        return shot

    approved_board = _resolve_approved_storyboard_board(
        ref_group=ref_group,
        ref_pack=ref_pack,
        approved_ref_id=approved_ref_id,
    )
    take_request = shot.take_request.model_copy(
        update={
            "approved_board": approved_board,
            "status": _resolve_take_request_status(shot.take_request),
        }
    )

    return shot.model_copy(
        update={
            "take_request": take_request,
            "status": _resolve_shot_status(shot),
        }
    )


def _resolve_approved_storyboard_board(
    *,
    ref_group: ShotReferenceGroup,
    ref_pack: RefPackV1,
    approved_ref_id: str,
) -> ApprovedStoryboardBoard:
    candidate = next(
        (
            candidate
            for candidate in ref_group.reference_candidates
            if candidate.ref_id == approved_ref_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError(
            f'Unknown approved ref_id "{approved_ref_id}" for shot "{ref_group.shot_id}"'
        )

    return ApprovedStoryboardBoard(
        ref_id=candidate.ref_id,
        asset_path=candidate.asset_path,
        source_ref_pack_id=ref_pack.ref_pack_id,
        source_review_payload_id=ref_pack.lineage.source_review_payload_id,
        first_frame_prompt_id=candidate.params.first_frame_prompt_id,
    )


def _resolve_take_request_status(take_request: ShotTakeRequest) -> TakeRequestStatus:
    if take_request.status == TakeRequestStatus.GENERATED:
        return take_request.status

    if take_request.generation_defaults.requires_approved_reference:
        return TakeRequestStatus.READY

    return take_request.status


def _resolve_shot_status(shot: ShotSpec) -> ShotStatus:
    if shot.status in {ShotStatus.GENERATED, ShotStatus.REVIEWED}:
        return shot.status

    if shot.take_request.generation_defaults.requires_approved_reference:
        return ShotStatus.READY_FOR_TAKES

    return shot.status
