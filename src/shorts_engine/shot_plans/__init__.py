"""Preset-driven shot plan contracts and compiler."""

from shorts_engine.shot_plans.compiler import compile_shot_plan
from shorts_engine.shot_plans.contracts import (
    BrandRuntimeInput,
    CompiledShotPlan,
    PresetShotTemplate,
    PresetSpec,
    PresetVersion,
    ProductConceptInput,
    ReferenceRequirement,
    ShotPlan,
    ShotSpec,
    ShotStatus,
    ShotTakeRequest,
    TakeGenerationDefaults,
    TakeRequestStatus,
)
from shorts_engine.shot_plans.presets import (
    FLAGSHIP_PRESET_ID,
    FLAGSHIP_PRESET_VERSION,
    PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT,
    get_shot_plan_preset,
    list_shot_plan_presets,
)

__all__ = [
    "FLAGSHIP_PRESET_ID",
    "FLAGSHIP_PRESET_VERSION",
    "PREMIUM_PRODUCT_MACRO_REVEAL_PACKSHOT",
    "BrandRuntimeInput",
    "CompiledShotPlan",
    "PresetShotTemplate",
    "PresetSpec",
    "PresetVersion",
    "ProductConceptInput",
    "ReferenceRequirement",
    "ShotPlan",
    "ShotSpec",
    "ShotStatus",
    "ShotTakeRequest",
    "TakeGenerationDefaults",
    "TakeRequestStatus",
    "compile_shot_plan",
    "get_shot_plan_preset",
    "list_shot_plan_presets",
]
