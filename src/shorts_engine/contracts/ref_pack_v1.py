"""Typed reference-pack artifact contract for shot-based Shortform V1."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

REF_PACK_SCHEMA_VERSION = "ref_pack.v1"


class ContractModel(BaseModel):
    """Base model for stable machine-readable contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReferenceCandidateStatus(StrEnum):
    """Lifecycle status for a generated reference candidate."""

    GENERATED = "generated"
    FAILED = "failed"


class ReferenceCandidateParams(ContractModel):
    """Stable generation inputs for one reference candidate."""

    aspect_ratio: str = Field(default="9:16", min_length=1)
    candidate_index: int = Field(ge=1)
    prompt: str = Field(min_length=1)
    first_frame_prompt_id: str = Field(min_length=1)
    reference_asset_ids: list[str] = Field(default_factory=list)
    provider_params: dict[str, Any] = Field(default_factory=dict)


class ReferenceCandidate(ContractModel):
    """Machine-readable metadata for one generated shot reference."""

    ref_id: str = Field(min_length=1)
    asset_path: str = Field(min_length=1)
    prompt_summary: str = Field(min_length=1)
    model: str = Field(min_length=1)
    params: ReferenceCandidateParams
    created_at: datetime
    status: ReferenceCandidateStatus = ReferenceCandidateStatus.GENERATED


class RefPackLineage(ContractModel):
    """Stable lineage metadata that ties a ref-pack to its review payload."""

    preset_id: str = Field(min_length=1)
    preset_version: str = Field(min_length=1)
    source_plan_id: str = Field(min_length=1)
    source_review_payload_id: str = Field(min_length=1)
    aspect_ratio: str = Field(default="9:16", min_length=1)
    reference_asset_ids: list[str] = Field(default_factory=list)
    review_guidance: list[str] = Field(default_factory=list)


class ShotReferenceGroup(ContractModel):
    """Grouped reference candidates for one shot."""

    shot_id: str = Field(min_length=1)
    reference_candidates: list[ReferenceCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference_candidates(self) -> ShotReferenceGroup:
        """Require unique ref ids within one shot group."""
        ref_ids = [candidate.ref_id for candidate in self.reference_candidates]
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError(f"Reference candidate ids must be unique within {self.shot_id}")
        return self


class RefPackV1(ContractModel):
    """Typed artifact emitted after still-reference generation."""

    ref_pack_id: str = Field(min_length=1)
    schema_version: str = Field(default=REF_PACK_SCHEMA_VERSION)
    job_id: str = Field(min_length=1)
    preset_id: str = Field(min_length=1)
    source_shot_plan_id: str = Field(min_length=1)
    lineage: RefPackLineage
    shots: list[ShotReferenceGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shots(self) -> RefPackV1:
        """Require stable shot grouping and globally unique ref ids."""
        if self.lineage.preset_id != self.preset_id:
            raise ValueError("Ref-pack lineage preset_id must match artifact preset_id")
        if self.lineage.source_plan_id != self.source_shot_plan_id:
            raise ValueError("Ref-pack lineage source_plan_id must match source_shot_plan_id")

        shot_ids = [shot.shot_id for shot in self.shots]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("Ref-pack shot ids must be unique")

        ref_ids = [
            candidate.ref_id
            for shot in self.shots
            for candidate in shot.reference_candidates
        ]
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("Ref-pack reference candidate ids must be globally unique")

        known_reference_asset_ids = set(self.lineage.reference_asset_ids)
        for shot in self.shots:
            for candidate in shot.reference_candidates:
                unknown_reference_asset_ids = (
                    set(candidate.params.reference_asset_ids) - known_reference_asset_ids
                )
                if unknown_reference_asset_ids:
                    unknown = ", ".join(sorted(unknown_reference_asset_ids))
                    raise ValueError(
                        f"Unknown ref-pack lineage reference asset ids for {candidate.ref_id}: {unknown}"
                    )

        return self
