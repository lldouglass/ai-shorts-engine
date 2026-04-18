"""Shot-plan preset registry."""

from __future__ import annotations

from shorts_engine.shot_plans.contracts import (
    PresetShotTemplate,
    PresetSpec,
    PresetVersion,
    ReferenceRequirement,
    TakeGenerationDefaults,
)

FLAGSHIP_PRESET_ID = "premium_product_macro_reveal_packshot"
FLAGSHIP_PRESET_VERSION = "1.0.0"


PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT = PresetSpec(
    identity=PresetVersion(
        preset_id=FLAGSHIP_PRESET_ID,
        version=FLAGSHIP_PRESET_VERSION,
    ),
    display_name="Premium Product Macro Reveal Packshot",
    description=(
        "A three-shot premium shortform product-ad structure: sensory macro hook, "
        "reveal/demo, then packshot payoff."
    ),
    aspect_ratio="9:16",
    runtime_target_seconds=8.0,
    shot_templates=[
        PresetShotTemplate(
            template_id="macro_hook",
            sequence_order=1,
            intent="Stop-scroll sensory hook that makes the product feel premium.",
            role="macro_hook",
            subject_template="{product_name} macro detail, emphasizing {primary_sensory_cue}",
            environment_template=(
                "{environment}; controlled tabletop setup with premium material contrast"
            ),
            motion_beat_template=(
                "Slow macro drift across the product surface, ending on a tactile detail"
            ),
            camera_language=(
                "Extreme close macro, shallow depth of field, crisp specular highlights"
            ),
            duration_target_seconds=2.0,
            reference_requirements=[
                ReferenceRequirement(
                    role="macro_reference",
                    description=(
                        "Close visual reference for {product_name} material, texture, "
                        "finish, and premium sensory cue."
                    ),
                    count=3,
                    approval_required=True,
                )
            ],
            take_generation_defaults=TakeGenerationDefaults(
                target_take_count=3,
                variation_axes=[
                    "micro camera drift direction",
                    "highlight placement",
                    "macro focal detail",
                ],
                notes=[
                    "Keep the product identity legible even at macro scale.",
                    "Avoid readable text, labels, or UI overlays.",
                ],
            ),
            variation_hints=[
                "Try one take with a left-to-right drift.",
                "Try one take with a slow push toward the hero texture.",
                "Try one take with the light catching the product edge.",
            ],
        ),
        PresetShotTemplate(
            template_id="reveal_demo",
            sequence_order=2,
            intent="Reveal the full product and demonstrate the core benefit.",
            role="reveal_demo",
            subject_template="{product_name} in use, clearly showing {key_benefit}",
            environment_template="{environment}; premium demo setup for {audience}",
            motion_beat_template=(
                "Product enters or rotates into view, then performs the simple demo beat"
            ),
            camera_language=(
                "Medium close product reveal, smooth push-in, clean parallax, realistic handoff"
            ),
            duration_target_seconds=3.0,
            reference_requirements=[
                ReferenceRequirement(
                    role="demo_reference",
                    description=(
                        "Reference showing the product form factor, demo action, and "
                        "how {key_benefit} should read visually."
                    ),
                    count=3,
                    approval_required=True,
                )
            ],
            take_generation_defaults=TakeGenerationDefaults(
                target_take_count=3,
                variation_axes=[
                    "demo timing",
                    "product orientation",
                    "camera push intensity",
                ],
                notes=[
                    "The demo should be visually understandable without captions.",
                    "Keep motion simple enough for review and winner selection.",
                ],
            ),
            variation_hints=[
                "Try one take with a clean hand interaction.",
                "Try one take with the product rotating into the benefit moment.",
                "Try one take with the benefit revealed by lighting change.",
            ],
        ),
        PresetShotTemplate(
            template_id="packshot_payoff",
            sequence_order=3,
            intent="Land the product memory with a final hero packshot payoff.",
            role="packshot_payoff",
            subject_template=("Hero packshot of {product_name}, anchored by {supporting_detail}"),
            environment_template=(
                "{environment}; clean premium end-frame with negative space for review notes"
            ),
            motion_beat_template="Subtle hero push-in, then hold steady for the payoff.",
            camera_language=(
                "Locked hero product angle, polished commercial lighting, stable final frame"
            ),
            duration_target_seconds=3.0,
            reference_requirements=[
                ReferenceRequirement(
                    role="packshot_reference",
                    description=(
                        "Approved hero product reference for final framing, silhouette, "
                        "color, and brand-safe finish."
                    ),
                    count=3,
                    approval_required=True,
                )
            ],
            take_generation_defaults=TakeGenerationDefaults(
                target_take_count=3,
                variation_axes=[
                    "hero angle",
                    "background depth",
                    "payoff hold timing",
                ],
                notes=[
                    "Hold the final product identity cleanly for review.",
                    "Leave the shot uncluttered for downstream edit decisions.",
                ],
            ),
            variation_hints=[
                "Try one take with a straight-on hero angle.",
                "Try one take with a three-quarter premium product angle.",
                "Try one take with a slower final push and longer hold.",
            ],
        ),
    ],
)

SHOT_PLAN_PRESETS: dict[tuple[str, str], PresetSpec] = {
    (
        PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT.preset_id,
        PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT.version,
    ): PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT
}


def get_shot_plan_preset(preset_id: str, version: str | None = None) -> PresetSpec | None:
    """Return a shot-plan preset by identity and version."""
    if version is None and preset_id == FLAGSHIP_PRESET_ID:
        version = FLAGSHIP_PRESET_VERSION

    return SHOT_PLAN_PRESETS.get((preset_id, version or ""))


def list_shot_plan_presets() -> list[PresetSpec]:
    """Return all registered shot-plan presets."""
    return list(SHOT_PLAN_PRESETS.values())
