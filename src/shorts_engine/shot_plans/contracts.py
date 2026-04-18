"""Typed contracts for preset-driven shot plan compilation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHOT_PLAN_SCHEMA_VERSION = "shot-plan.v1"


class ShotStatus(StrEnum):
    """Lifecycle status for a compiled shot before generation."""

    PLANNED = "planned"
    NEEDS_REFERENCES = "needs_references"
    READY_FOR_TAKES = "ready_for_takes"
    GENERATED = "generated"
    REVIEWED = "reviewed"


class TakeRequestStatus(StrEnum):
    """Lifecycle status for a take-generation request."""

    REQUESTED = "requested"
    BLOCKED_ON_REFERENCES = "blocked_on_references"
    READY = "ready"
    GENERATED = "generated"


class ContractModel(BaseModel):
    """Base model for stable JSON handoff contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PresetVersion(ContractModel):
    """Identity for a versioned shot-plan preset."""

    preset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    schema_version: str = Field(default=SHOT_PLAN_SCHEMA_VERSION)


class ReferenceRequirement(ContractModel):
    """Reference asset requirement for one shot."""

    role: str = Field(min_length=1)
    description: str = Field(min_length=1)
    count: int = Field(default=3, ge=1)
    approval_required: bool = True
    required_before: Literal["take_generation", "final_review"] = "take_generation"


class TakeGenerationDefaults(ContractModel):
    """Generic take-generation defaults, intentionally provider-neutral."""

    target_take_count: int = Field(default=3, ge=1)
    seed_policy: str = Field(default="deterministic_per_shot_take", min_length=1)
    variation_strategy: str = Field(default="controlled_small_variations", min_length=1)
    variation_axes: list[str] = Field(default_factory=list)
    avoid_visible_text: bool = True
    requires_approved_reference: bool = True
    notes: list[str] = Field(default_factory=list)


class PresetShotTemplate(ContractModel):
    """Reusable shot template inside a versioned preset."""

    template_id: str = Field(min_length=1)
    sequence_order: int = Field(ge=1)
    intent: str = Field(min_length=1)
    role: str = Field(min_length=1)
    subject_template: str = Field(min_length=1)
    environment_template: str = Field(min_length=1)
    motion_beat_template: str = Field(min_length=1)
    camera_language: str = Field(min_length=1)
    duration_target_seconds: float = Field(gt=0)
    reference_requirements: list[ReferenceRequirement] = Field(min_length=1)
    take_generation_defaults: TakeGenerationDefaults = Field(default_factory=TakeGenerationDefaults)
    variation_hints: list[str] = Field(default_factory=list)


class PresetSpec(ContractModel):
    """First-class generator-side preset contract."""

    identity: PresetVersion
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    aspect_ratio: str = Field(default="9:16", min_length=1)
    runtime_target_seconds: float = Field(gt=0)
    shot_templates: list[PresetShotTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shot_templates(self) -> PresetSpec:
        """Require unique shot ids and sequence positions inside a preset."""
        template_ids = [shot.template_id for shot in self.shot_templates]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("Preset shot template ids must be unique")

        sequence_orders = [shot.sequence_order for shot in self.shot_templates]
        if len(sequence_orders) != len(set(sequence_orders)):
            raise ValueError("Preset shot sequence orders must be unique")

        expected_orders = list(range(1, len(sequence_orders) + 1))
        if sorted(sequence_orders) != expected_orders:
            raise ValueError("Preset shot sequence orders must be contiguous from 1")

        return self

    @property
    def preset_id(self) -> str:
        """Convenience accessor for preset id."""
        return self.identity.preset_id

    @property
    def version(self) -> str:
        """Convenience accessor for preset version."""
        return self.identity.version


class ProductConceptInput(ContractModel):
    """Product or concept inputs used by the deterministic shot-plan compiler."""

    product_name: str = Field(default="the product", min_length=1)
    product_category: str | None = None
    concept: str | None = None
    key_benefit: str = Field(default="a clear premium payoff", min_length=1)
    audience: str | None = None
    primary_sensory_cue: str | None = None
    supporting_detail: str | None = None
    use_case: str | None = None
    visual_constraints: list[str] = Field(default_factory=list)


class BrandRuntimeInput(ContractModel):
    """Optional brand and runtime inputs for shot-plan compilation."""

    brand_name: str | None = None
    brand_voice: str | None = None
    visual_style: str | None = None
    environment: str | None = None
    runtime_target_seconds: float | None = Field(default=None, gt=0)


class ShotTakeRequest(ContractModel):
    """Downstream-ready request metadata for generating takes for one shot."""

    take_request_id: str = Field(min_length=1)
    shot_id: str = Field(min_length=1)
    preset_id: str = Field(min_length=1)
    preset_version: str = Field(min_length=1)
    sequence_order: int = Field(ge=1)
    intent: str = Field(min_length=1)
    role: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    motion_beat: str = Field(min_length=1)
    camera_language: str = Field(min_length=1)
    duration_target_seconds: float = Field(gt=0)
    reference_requirements: list[ReferenceRequirement] = Field(min_length=1)
    generation_defaults: TakeGenerationDefaults
    variation_hints: list[str] = Field(default_factory=list)
    status: TakeRequestStatus = TakeRequestStatus.BLOCKED_ON_REFERENCES
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShotSpec(ContractModel):
    """Compiled shot spec ready for references, takes, and review."""

    shot_id: str = Field(min_length=1)
    sequence_order: int = Field(ge=1)
    intent: str = Field(min_length=1)
    role: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    motion_beat: str = Field(min_length=1)
    camera_language: str = Field(min_length=1)
    duration_target_seconds: float = Field(gt=0)
    reference_requirements: list[ReferenceRequirement] = Field(min_length=1)
    take_generation_defaults: TakeGenerationDefaults
    variation_hints: list[str] = Field(default_factory=list)
    take_request: ShotTakeRequest
    status: ShotStatus = ShotStatus.NEEDS_REFERENCES


class CompiledShotPlan(ContractModel):
    """Typed multi-shot package emitted by the preset compiler."""

    plan_id: str = Field(min_length=1)
    schema_version: str = Field(default=SHOT_PLAN_SCHEMA_VERSION)
    preset: PresetVersion
    product: ProductConceptInput
    brand: BrandRuntimeInput | None = None
    runtime_target_seconds: float = Field(gt=0)
    compiler_version: str = Field(default="shot-plan-compiler.v1", min_length=1)
    shots: list[ShotSpec] = Field(min_length=1)
    status: ShotStatus = ShotStatus.PLANNED

    @model_validator(mode="after")
    def validate_shots(self) -> CompiledShotPlan:
        """Require stable shot ordering in the compiled plan."""
        shot_ids = [shot.shot_id for shot in self.shots]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("Compiled shot ids must be unique")

        sequence_orders = [shot.sequence_order for shot in self.shots]
        if sorted(sequence_orders) != list(range(1, len(sequence_orders) + 1)):
            raise ValueError("Compiled shot sequence orders must be contiguous from 1")

        return self

    @property
    def shot_count(self) -> int:
        """Number of shots in this plan."""
        return len(self.shots)


ShotPlan = CompiledShotPlan
