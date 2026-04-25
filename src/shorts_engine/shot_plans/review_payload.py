"""Provider-neutral first-frame review payload builder."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from shorts_engine.shot_plans.contracts import (
    CompiledShotPlan,
    FirstFramePromptInputs,
    FirstFrameReferenceAsset,
    FirstFrameReviewPayload,
    FirstFrameReviewShot,
    ReferenceRequirement,
)

DEFAULT_FIRST_FRAME_REVIEW_GUIDANCE = [
    "Review and select one approved storyboard board / first-frame direction per shot before motion.",
    "Each still should read like a designed beat-deck board with one clear beat and short readable copy.",
    "Keep the same locked hero subject/object set, visual world, and layout system across the full sequence.",
    "Reject any still that violates product, packaging, mascot, audience, or claim constraints.",
    "Take generation remains blocked until the approved storyboard board/reference is selected.",
]


def build_first_frame_review_payload(
    plan: CompiledShotPlan,
    *,
    reference_assets: Sequence[FirstFrameReferenceAsset | Mapping[str, Any]] | None = None,
    aspect_ratio: str = "9:16",
    review_guidance: Sequence[str] | None = None,
) -> FirstFrameReviewPayload:
    """Build a deterministic first-frame review package from a compiled shot plan."""
    assets = [_coerce_reference_asset(asset) for asset in reference_assets or []]
    guidance = list(review_guidance or DEFAULT_FIRST_FRAME_REVIEW_GUIDANCE)
    reference_asset_ids = [asset.asset_id for asset in assets]

    shots: list[FirstFrameReviewShot] = []
    for shot in sorted(plan.shots, key=lambda item: item.sequence_order):
        prompt_inputs = FirstFramePromptInputs(
            aspect_ratio=aspect_ratio,
            product_name=plan.product.product_name,
            brand_name=plan.brand.brand_name if plan.brand else None,
            shot_id=shot.shot_id,
            sequence_order=shot.sequence_order,
            role=shot.role,
            intent=shot.intent,
            subject=shot.subject,
            environment=shot.environment,
            camera_language=shot.camera_language,
            storyboard_deck=plan.storyboard_deck,
            storyboard_board=shot.storyboard_board,
            motion_beat_after_approval=shot.motion_beat,
            preserve_approved_board_text=shot.take_generation_defaults.preserve_approved_board_text,
            visual_constraints=plan.product.visual_constraints,
            reference_asset_ids=reference_asset_ids,
        )
        shots.append(
            FirstFrameReviewShot(
                shot_id=shot.shot_id,
                sequence_order=shot.sequence_order,
                role=shot.role,
                intent=shot.intent,
                duration_target_seconds=shot.duration_target_seconds,
                storyboard_deck=plan.storyboard_deck,
                storyboard_board=shot.storyboard_board,
                reference_requirements=shot.reference_requirements,
                reference_asset_ids=reference_asset_ids,
                first_frame_prompt_id=f"{shot.shot_id}_first_frame_prompt",
                prompt_inputs=prompt_inputs,
                review_prompt_text=_build_review_prompt_text(
                    plan=plan,
                    prompt_inputs=prompt_inputs,
                    reference_assets=assets,
                    reference_requirements=shot.reference_requirements,
                    review_guidance=guidance,
                ),
            )
        )

    return FirstFrameReviewPayload(
        payload_id=_payload_id(
            plan=plan,
            reference_assets=assets,
            aspect_ratio=aspect_ratio,
            review_guidance=guidance,
        ),
        source_plan_id=plan.plan_id,
        preset=plan.preset,
        product=plan.product,
        brand=plan.brand,
        runtime_target_seconds=plan.runtime_target_seconds,
        storyboard_deck=plan.storyboard_deck,
        aspect_ratio=aspect_ratio,
        reference_assets=assets,
        review_guidance=guidance,
        shots=shots,
    )


def _coerce_reference_asset(
    asset: FirstFrameReferenceAsset | Mapping[str, Any],
) -> FirstFrameReferenceAsset:
    if isinstance(asset, FirstFrameReferenceAsset):
        return asset

    return FirstFrameReferenceAsset.model_validate(dict(asset))


def _build_review_prompt_text(
    *,
    plan: CompiledShotPlan,
    prompt_inputs: FirstFramePromptInputs,
    reference_assets: Sequence[FirstFrameReferenceAsset],
    reference_requirements: Sequence[ReferenceRequirement],
    review_guidance: Sequence[str],
) -> str:
    brand_name = prompt_inputs.brand_name or "the brand"
    lines = [
        (
            f"Create a vertical {prompt_inputs.aspect_ratio} storyboard board still for review "
            "before any motion or take generation."
        ),
        f"Shot {prompt_inputs.sequence_order} ({prompt_inputs.role}): {prompt_inputs.intent}",
        f"Product: {prompt_inputs.product_name}",
        f"Brand: {brand_name}",
        f"Sequence visual world: {prompt_inputs.storyboard_deck.visual_world}",
        f"Sequence layout system: {prompt_inputs.storyboard_deck.layout_system}",
        f"Copy style: {prompt_inputs.storyboard_deck.copy_style}",
        f"Subject direction: {prompt_inputs.subject}",
        f"Environment direction: {prompt_inputs.environment}",
        f"Camera language: {prompt_inputs.camera_language}",
        f"Board layout notes: {prompt_inputs.storyboard_board.layout_notes}",
        f"This still should set up this later motion only after approval: "
        f"{prompt_inputs.motion_beat_after_approval}",
    ]
    lines.append("Sequence continuity locks:")
    lines.extend(f"- {lock}" for lock in prompt_inputs.storyboard_deck.continuity_locks)

    if prompt_inputs.storyboard_board.title:
        lines.append(f"Board title: {prompt_inputs.storyboard_board.title}")
    if prompt_inputs.storyboard_board.hook_role:
        lines.append(f"Board hook role: {prompt_inputs.storyboard_board.hook_role}")
    if prompt_inputs.storyboard_board.on_frame_text:
        lines.append(
            f'Readable on-frame board copy: "{prompt_inputs.storyboard_board.on_frame_text}"'
        )
    lines.append("Treat this as one clear designed beat-deck board, not a generic clip frame.")
    if (
        prompt_inputs.preserve_approved_board_text
        and prompt_inputs.storyboard_board.on_frame_text
    ):
        lines.append(
            "If approved later for motion, preserve the approved board copy and designed layout."
        )

    if plan.product.key_benefit:
        lines.append(f"Benefit angle: {plan.product.key_benefit}")
    if plan.product.audience:
        lines.append(f"Audience: {plan.product.audience}")

    if prompt_inputs.visual_constraints:
        lines.append("Product and brand locks:")
        lines.extend(f"- {constraint}" for constraint in prompt_inputs.visual_constraints)

    if reference_assets:
        lines.append("Sequence lock reference assets to preserve across every board:")
        lines.extend(
            f"- {asset.asset_id} ({asset.role}): {asset.description} [{asset.uri}]"
            for asset in reference_assets
            if asset.required
        )

    lines.append("Shot reference requirements:")
    lines.extend(
        f"- {requirement.role}: {requirement.description}"
        for requirement in reference_requirements
    )

    lines.append("Review gate:")
    lines.extend(f"- {item}" for item in review_guidance)
    lines.append("Do not create video, animation, motion, or take variations yet.")

    return "\n".join(lines)


def _payload_id(
    *,
    plan: CompiledShotPlan,
    reference_assets: Sequence[FirstFrameReferenceAsset],
    aspect_ratio: str,
    review_guidance: Sequence[str],
) -> str:
    payload = {
        "plan_id": plan.plan_id,
        "reference_assets": [asset.model_dump(mode="json") for asset in reference_assets],
        "aspect_ratio": aspect_ratio,
        "review_guidance": list(review_guidance),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"firstframe_{_slug(plan.preset.preset_id)}_{digest}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"
