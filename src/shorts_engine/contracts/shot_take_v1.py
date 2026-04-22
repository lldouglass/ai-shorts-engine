"""Typed take artifact contract for shot-based Shortform V1."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHOT_TAKE_SCHEMA_VERSION = "shot_take.v1"


class ContractModel(BaseModel):
    """Base model for stable machine-readable contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShotTakeStatus(StrEnum):
    """Lifecycle status for a generated take artifact."""

    GENERATED = "generated"
    FAILED = "failed"


class ShotTakeParams(ContractModel):
    """Stable generation inputs captured for one take."""

    aspect_ratio: str = Field(default="9:16", min_length=1)
    duration_seconds: float = Field(gt=0)
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = Field(default=None, min_length=1)
    take_index: int = Field(ge=1)
    reference_asset_paths: list[str] = Field(default_factory=list)
    variation_hint: str | None = Field(default=None, min_length=1)
    provider_params: dict[str, Any] = Field(default_factory=dict)


class ShotTakeLineage(ContractModel):
    """Stable lineage metadata that ties a take to its source request."""

    preset_id: str = Field(min_length=1)
    preset_version: str = Field(min_length=1)
    sequence_order: int = Field(ge=1)
    source_plan_id: str | None = Field(default=None, min_length=1)
    reference_asset_paths: list[str] = Field(default_factory=list)
    provider_job_id: str | None = Field(default=None, min_length=1)
    request_metadata: dict[str, Any] = Field(default_factory=dict)


class ShotTakeArtifact(ContractModel):
    """Machine-readable metadata for one generated shot take."""

    take_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    shot_id: str = Field(min_length=1)
    take_request_id: str = Field(min_length=1)
    asset_path: str | None = Field(default=None, min_length=1)
    model: str = Field(min_length=1)
    params: ShotTakeParams
    seed: int | None = Field(default=None, ge=0)
    cost_estimate: float | None = Field(default=None, ge=0)
    created_at: datetime
    status: ShotTakeStatus
    error_message: str | None = Field(default=None, min_length=1)
    lineage: ShotTakeLineage

    @model_validator(mode="after")
    def validate_status_fields(self) -> ShotTakeArtifact:
        """Require generated takes to carry an asset and failed takes to carry an error."""
        if self.status == ShotTakeStatus.GENERATED and not self.asset_path:
            raise ValueError("Generated takes must include an asset_path")

        if self.status == ShotTakeStatus.FAILED and not self.error_message:
            raise ValueError("Failed takes must include an error_message")

        return self


class ShotTakeBatchV1(ContractModel):
    """Typed take artifact emitted for one shot take request."""

    take_batch_id: str = Field(min_length=1)
    schema_version: str = Field(default=SHOT_TAKE_SCHEMA_VERSION)
    job_id: str = Field(min_length=1)
    shot_id: str = Field(min_length=1)
    take_request_id: str = Field(min_length=1)
    takes: list[ShotTakeArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_takes(self) -> ShotTakeBatchV1:
        """Require stable batch identity and unique take ids."""
        take_ids = [take.take_id for take in self.takes]
        if len(take_ids) != len(set(take_ids)):
            raise ValueError("Shot take ids must be unique within a batch")

        for take in self.takes:
            if take.job_id != self.job_id:
                raise ValueError("Shot take batch job_id must match each take job_id")
            if take.shot_id != self.shot_id:
                raise ValueError("Shot take batch shot_id must match each take shot_id")
            if take.take_request_id != self.take_request_id:
                raise ValueError(
                    "Shot take batch take_request_id must match each take take_request_id"
                )

        return self
