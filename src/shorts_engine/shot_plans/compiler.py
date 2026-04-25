"""Deterministic compiler for preset-driven shot plans."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from shorts_engine.shot_plans.contracts import (
    BrandRuntimeInput,
    CompiledShotPlan,
    PresetSpec,
    ProductConceptInput,
    ReferenceRequirement,
    ShotSpec,
    ShotTakeRequest,
    StoryboardBoardSpec,
    StoryboardDeckSpec,
    TakeGenerationDefaults,
)
from shorts_engine.shot_plans.presets import get_shot_plan_preset


def compile_shot_plan(
    preset_id: str,
    preset_version: str,
    product: ProductConceptInput | Mapping[str, Any] | None = None,
    *,
    concept: str | None = None,
    brand: BrandRuntimeInput | Mapping[str, Any] | None = None,
) -> CompiledShotPlan:
    """Compile a versioned preset and product/concept inputs into a shot plan.

    The compiler is intentionally deterministic: it does not call an LLM, does not
    stamp the current time, and derives plan/shot/take ids from normalized inputs.
    """
    preset = get_shot_plan_preset(preset_id, preset_version)
    if preset is None:
        raise ValueError(f"Unknown shot-plan preset: {preset_id}@{preset_version}")

    product_input = _coerce_product(product, concept)
    brand_input = _coerce_brand(brand)
    runtime_target_seconds = brand_input.runtime_target_seconds or preset.runtime_target_seconds
    context = _build_template_context(product_input, brand_input)
    plan_id = _plan_id(preset, product_input, brand_input)
    durations = _scaled_durations(
        [template.duration_target_seconds for template in preset.shot_templates],
        runtime_target_seconds,
    )
    storyboard_deck = _render_storyboard_deck(preset.storyboard_deck, context)

    shots: list[ShotSpec] = []
    for template, duration_target_seconds in zip(
        sorted(preset.shot_templates, key=lambda shot: shot.sequence_order),
        durations,
        strict=True,
    ):
        shot_id = _shot_id(
            preset_id=preset.preset_id,
            preset_version=preset.version,
            sequence_order=template.sequence_order,
            role=template.role,
        )
        reference_requirements = [
            _render_reference_requirement(requirement, context)
            for requirement in template.reference_requirements
        ]
        generation_defaults = _compile_take_defaults(template.take_generation_defaults)
        variation_hints = [_render_template(hint, context) for hint in template.variation_hints]
        subject = _render_template(template.subject_template, context)
        environment = _render_template(template.environment_template, context)
        motion_beat = _render_template(template.motion_beat_template, context)
        camera_language = template.camera_language
        storyboard_board = _render_storyboard_board(template.storyboard_board, context)

        take_request = ShotTakeRequest(
            take_request_id=f"{shot_id}_take_request",
            shot_id=shot_id,
            preset_id=preset.preset_id,
            preset_version=preset.version,
            sequence_order=template.sequence_order,
            intent=template.intent,
            role=template.role,
            subject=subject,
            environment=environment,
            motion_beat=motion_beat,
            camera_language=camera_language,
            duration_target_seconds=duration_target_seconds,
            storyboard_deck=storyboard_deck,
            storyboard_board=storyboard_board,
            reference_requirements=reference_requirements,
            generation_defaults=generation_defaults,
            variation_hints=variation_hints,
            metadata={
                "intent": template.intent,
                "product_name": product_input.product_name,
                "concept": product_input.concept,
                "requires_review": True,
                "source_plan_id": plan_id,
            },
        )

        shots.append(
            ShotSpec(
                shot_id=shot_id,
                sequence_order=template.sequence_order,
                intent=template.intent,
                role=template.role,
                subject=subject,
                environment=environment,
                motion_beat=motion_beat,
                camera_language=camera_language,
                duration_target_seconds=duration_target_seconds,
                storyboard_deck=storyboard_deck,
                storyboard_board=storyboard_board,
                reference_requirements=reference_requirements,
                take_generation_defaults=generation_defaults,
                variation_hints=variation_hints,
                take_request=take_request,
            )
        )

    return CompiledShotPlan(
        plan_id=plan_id,
        preset=preset.identity,
        product=product_input,
        brand=brand_input,
        runtime_target_seconds=runtime_target_seconds,
        storyboard_deck=storyboard_deck,
        shots=shots,
    )


def _coerce_product(
    product: ProductConceptInput | Mapping[str, Any] | None,
    concept: str | None,
) -> ProductConceptInput:
    if isinstance(product, ProductConceptInput):
        product_input = product
    else:
        product_input = ProductConceptInput.model_validate(dict(product or {}))

    if concept and product_input.concept != concept:
        product_input = product_input.model_copy(update={"concept": concept})

    return product_input


def _coerce_brand(brand: BrandRuntimeInput | Mapping[str, Any] | None) -> BrandRuntimeInput:
    if isinstance(brand, BrandRuntimeInput):
        return brand

    return BrandRuntimeInput.model_validate(dict(brand or {}))


def _build_template_context(
    product: ProductConceptInput,
    brand: BrandRuntimeInput,
) -> dict[str, str]:
    key_benefit = product.key_benefit
    primary_sensory_cue = product.primary_sensory_cue or "material texture, finish, and light"
    audience = product.audience or "quality-conscious shortform shoppers"
    environment = brand.environment or "minimal premium studio environment"
    supporting_detail = product.supporting_detail or key_benefit
    concept = product.concept or f"{product.product_name} premium product short"
    benefit_copy = _short_copy(key_benefit, limit=56)
    sensory_copy = _short_copy(primary_sensory_cue, limit=56)
    use_case_copy = _short_copy(product.use_case or key_benefit, limit=56)
    product_category_copy = _short_copy(
        product.product_category or product.product_name,
        limit=48,
    )
    supporting_detail_copy = _short_copy(supporting_detail, limit=56)
    brand_display_name = brand.brand_name or _short_copy(product.product_name, limit=40)

    return {
        "product_name": product.product_name,
        "product_category": product.product_category or "premium product",
        "concept": concept,
        "key_benefit": key_benefit,
        "audience": audience,
        "primary_sensory_cue": primary_sensory_cue,
        "supporting_detail": supporting_detail,
        "use_case": product.use_case or "daily use",
        "brand_name": brand.brand_name or "the brand",
        "brand_display_name": brand_display_name,
        "brand_voice": brand.brand_voice or "premium and direct",
        "visual_style": brand.visual_style or "cinematic premium realism",
        "environment": environment,
        "benefit_copy": benefit_copy,
        "sensory_copy": sensory_copy,
        "use_case_copy": use_case_copy,
        "product_category_copy": product_category_copy,
        "supporting_detail_copy": supporting_detail_copy,
    }


def _render_reference_requirement(
    requirement: ReferenceRequirement,
    context: Mapping[str, str],
) -> ReferenceRequirement:
    return requirement.model_copy(
        update={"description": _render_template(requirement.description, context)}
    )


def _compile_take_defaults(defaults: TakeGenerationDefaults) -> TakeGenerationDefaults:
    return defaults.model_copy()


def _render_storyboard_deck(
    storyboard_deck: StoryboardDeckSpec,
    context: Mapping[str, str],
) -> StoryboardDeckSpec:
    return storyboard_deck.model_copy(
        update={
            "visual_world": _render_template(storyboard_deck.visual_world, context),
            "layout_system": _render_template(storyboard_deck.layout_system, context),
            "copy_style": _render_template(storyboard_deck.copy_style, context),
            "continuity_locks": [
                _render_template(lock, context) for lock in storyboard_deck.continuity_locks
            ],
        }
    )


def _render_storyboard_board(
    storyboard_board: StoryboardBoardSpec,
    context: Mapping[str, str],
) -> StoryboardBoardSpec:
    return storyboard_board.model_copy(
        update={
            "title": (
                _render_template(storyboard_board.title, context)
                if storyboard_board.title
                else None
            ),
            "hook_role": (
                _render_template(storyboard_board.hook_role, context)
                if storyboard_board.hook_role
                else None
            ),
            "on_frame_text": (
                _render_template(storyboard_board.on_frame_text, context)
                if storyboard_board.on_frame_text
                else None
            ),
            "layout_notes": _render_template(storyboard_board.layout_notes, context),
        }
    )


def _render_template(template: str, context: Mapping[str, str]) -> str:
    return template.format_map(dict(context))


def _scaled_durations(base_durations: list[float], runtime_target_seconds: float) -> list[float]:
    base_total = sum(base_durations)
    if base_total <= 0:
        raise ValueError("Preset duration total must be greater than zero")

    scaled = [
        round(duration * runtime_target_seconds / base_total, 2) for duration in base_durations
    ]
    if len(scaled) > 1:
        scaled[-1] = round(runtime_target_seconds - sum(scaled[:-1]), 2)
    return scaled


def _plan_id(
    preset: PresetSpec,
    product: ProductConceptInput,
    brand: BrandRuntimeInput,
) -> str:
    payload = {
        "preset_id": preset.preset_id,
        "preset_version": preset.version,
        "preset_spec": preset.model_dump(mode="json"),
        "product": product.model_dump(mode="json", exclude_none=True),
        "brand": brand.model_dump(mode="json", exclude_none=True),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"shotplan_{_slug(preset.preset_id)}_{_slug(preset.version)}_{digest}"


def _shot_id(
    *,
    preset_id: str,
    preset_version: str,
    sequence_order: int,
    role: str,
) -> str:
    return f"{_slug(preset_id)}_{_slug(preset_version)}_" f"s{sequence_order:02d}_{_slug(role)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def _short_copy(value: str, *, limit: int) -> str:
    cleaned = " ".join(value.split()).strip().rstrip(".")
    if len(cleaned) <= limit:
        return cleaned

    for separator in (" with ", ";", ".", ",", " but ", " for "):
        if separator not in cleaned:
            continue
        candidate = cleaned.split(separator, 1)[0].strip()
        if 8 <= len(candidate) <= limit:
            return candidate

    words: list[str] = []
    for word in cleaned.split():
        candidate = " ".join([*words, word])
        if len(candidate) > limit:
            break
        words.append(word)

    return " ".join(words) if words else cleaned[:limit].rstrip()
