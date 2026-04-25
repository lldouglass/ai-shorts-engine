"""Narrow benchmark fixtures for real product shot-package exports."""

from __future__ import annotations

from shorts_engine.shot_plans.compiler import compile_shot_plan
from shorts_engine.shot_plans.contracts import (
    BrandRuntimeInput,
    CompiledShotPlan,
    FirstFrameReferenceAsset,
    FirstFrameReviewPayload,
    ProductConceptInput,
)
from shorts_engine.shot_plans.presets import FLAGSHIP_PRESET_ID, FLAGSHIP_PRESET_VERSION
from shorts_engine.shot_plans.review_payload import build_first_frame_review_payload

TALL_OWL_BENCHMARK_ID = "tall_owl_whipped_tallow_vanilla_orange"

TALL_OWL_PRODUCT_INPUT = ProductConceptInput(
    product_name="Tall Owl Whipped Tallow Vanilla Orange",
    product_category="whipped tallow skincare balm",
    concept=(
        "Premium paid-social skincare ad for Tall Owl Whipped Tallow Vanilla Orange, "
        "built for dry, sensitive, reactive skin."
    ),
    key_benefit=(
        "deep moisture for dry, sensitive, reactive skin with a whipped light, "
        "not greasy feel"
    ),
    audience="people with dry, sensitive, reactive skin who want a simpler routine",
    primary_sensory_cue="airy whipped white tallow texture in the Vanilla Orange jar",
    supporting_detail="the real Tall Owl owl logo mascot beside the faithful Vanilla Orange jar",
    use_case="simplifying a dry, reactive skincare routine",
    visual_constraints=[
        (
            "Use the real Tall Owl owl logo mark as the mascot and spokesperson; "
            "do not invent a new owl or use a different cartoon owl."
        ),
        (
            "Keep the actual Tall Owl Whipped Tallow Vanilla Orange jar packaging faithful: "
            "frosted glass jar, brushed silver ribbed metal lid, white label, thin orange "
            "lines near top and bottom, green and black owl logo, TALL OWL in black caps, "
            "Whipped in gray script, TALLOW in large black serif caps, VANILLA ORANGE in "
            "white caps on an orange brushstroke, Net Wt. 3.0 oz."
        ),
        "Lead with dry, sensitive, reactive skin; do not drift into generic luxury skincare.",
        "Texture should feel whipped, light, and airy; not waxy, heavy, or greasy.",
        "No people, no extra products, no fake branding, no packaging drift, no made-up text.",
        (
            "Avoid explicit medical treatment claims such as cures eczema, treats rosacea, "
            "fixes dermatitis, or before-after disease framing."
        ),
    ],
)

TALL_OWL_BRAND_INPUT = BrandRuntimeInput(
    brand_name="Tall Owl",
    brand_voice="simple, calm, direct, policy-safe",
    visual_style="clean premium skincare-ad realism with warm natural light",
    environment="bright clean premium skincare setup with safe copy space",
    runtime_target_seconds=8.0,
)

TALL_OWL_REFERENCE_ASSETS = [
    FirstFrameReferenceAsset(
        asset_id="tall_owl_vanilla_orange_packaging_front",
        role="product_packaging_lock",
        uri=(
            "/Users/openclaw/.openclaw/workspace/assets/tall-owl/vanilla-orange-refs/"
            "Whipped Tallow VANILLA ORANGE 3oz.4.png"
        ),
        description=(
            "Real Tall Owl Vanilla Orange 3.0 oz jar packaging reference from Jack's email; "
            "preserve jar shape, lid, label hierarchy, scent, logo, and readable product lock."
        ),
    ),
    FirstFrameReferenceAsset(
        asset_id="tall_owl_vanilla_orange_texture",
        role="product_texture_lock",
        uri=(
            "/Users/openclaw/.openclaw/workspace/assets/tall-owl/vanilla-orange-refs/"
            "Whipped Tallow VANILLA ORANGE 3oz.1.png"
        ),
        description=(
            "Real Tall Owl Vanilla Orange product reference for the whipped white tallow "
            "texture and frosted jar finish."
        ),
    ),
    FirstFrameReferenceAsset(
        asset_id="tall_owl_owl_mascot_hires",
        role="mascot_logo_lock",
        uri="/Users/openclaw/.openclaw/workspace/assets/tall-owl/owl-mascot-hires.png",
        description=(
            "Real Tall Owl owl logo mark to use as the mascot/spokesowl; do not replace "
            "it with a newly invented owl."
        ),
    ),
    FirstFrameReferenceAsset(
        asset_id="tall_owl_full_logo",
        role="brand_logo_lock",
        uri=(
            "/Users/openclaw/.openclaw/workspace/tmp/tall-owl-brand/"
            "TALL_OWL_FULL_LOGO_V2.psd_400_x_100_px_600_x_100_px_1.png"
        ),
        description="Real Tall Owl full logo reference for brand typography and owl placement.",
    ),
]

TALL_OWL_FIRST_FRAME_REVIEW_GUIDANCE = [
    "Select one approved storyboard board / first-frame direction per shot before any motion work.",
    (
        "Keep one clear beat per board with short readable copy, a consistent Tall Owl visual "
        "world, and the same locked jar/owl/logo subject set across the full sequence."
    ),
    "Reject stills with owl mascot drift, Vanilla Orange jar drift, fake text, or unreadable labels.",
    "Reject generic luxury skincare directions that lose the dry, sensitive, reactive skin angle.",
    "Take generation remains blocked until the approved storyboard board/reference is selected.",
]


def compile_tall_owl_benchmark_shot_plan() -> CompiledShotPlan:
    """Compile the Tall Owl benchmark through the existing flagship shot-plan preset."""
    return compile_shot_plan(
        FLAGSHIP_PRESET_ID,
        FLAGSHIP_PRESET_VERSION,
        product=TALL_OWL_PRODUCT_INPUT,
        brand=TALL_OWL_BRAND_INPUT,
    )


def build_tall_owl_first_frame_review_payload() -> FirstFrameReviewPayload:
    """Build the provider-neutral Tall Owl first-frame review export payload."""
    return build_first_frame_review_payload(
        compile_tall_owl_benchmark_shot_plan(),
        reference_assets=TALL_OWL_REFERENCE_ASSETS,
        review_guidance=TALL_OWL_FIRST_FRAME_REVIEW_GUIDANCE,
    )
