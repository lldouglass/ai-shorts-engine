"""Generate typed reference packs for shot-based Shortform V1."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from shorts_engine.config import settings
from shorts_engine.contracts.ref_pack_v1 import (
    ReferenceCandidate,
    ReferenceCandidateParams,
    RefPackLineage,
    RefPackV1,
    ShotReferenceGroup,
)
from shorts_engine.logging import get_logger
from shorts_engine.shot_plans.contracts import (
    CompiledShotPlan,
    FirstFrameReferenceAsset,
    FirstFrameReviewPayload,
    FirstFrameReviewShot,
)
from shorts_engine.shot_plans.review_payload import build_first_frame_review_payload

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class GeneratedReferenceImage:
    """Raw image bytes plus provider metadata from a reference generation call."""

    image_bytes: bytes
    mime_type: str
    provider_params: Mapping[str, Any] = field(default_factory=dict)


class ReferenceImageGenerator(Protocol):
    """Seam for the preferred reference-image route."""

    async def generate(
        self,
        prompt: str,
        *,
        candidate_index: int,
        aspect_ratio: str,
        model: str,
        shot: FirstFrameReviewShot,
        reference_assets: Sequence[FirstFrameReferenceAsset],
    ) -> GeneratedReferenceImage: ...


class GeminiNanoBananaReferenceImageGenerator:
    """Generate still references through the preferred Gemini Nano Banana route."""

    DEFAULT_MODEL = "nano-banana-pro-preview"
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key or settings.google_api_key
        self.model = model or self.DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        prompt: str,
        *,
        candidate_index: int,
        aspect_ratio: str,
        model: str,
        shot: FirstFrameReviewShot,
        reference_assets: Sequence[FirstFrameReferenceAsset],
    ) -> GeneratedReferenceImage:
        """Call the existing Nano Banana route without introducing a new provider layer."""
        if not self.api_key:
            raise ValueError("Google API key not configured for Nano Banana reference generation")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }
        url = f"{self.API_BASE_URL}/models/{model}:generateContent"

        logger.info(
            "ref_pack_reference_request_started",
            model=model,
            shot_id=shot.shot_id,
            candidate_index=candidate_index,
            aspect_ratio=aspect_ratio,
            reference_asset_count=len(reference_assets),
        )

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, params={"key": self.api_key}, json=payload)
            response.raise_for_status()

        image_part = _extract_inline_image_part(response.json())
        if image_part is None:
            raise RuntimeError(
                f"Nano Banana did not return an image for {shot.shot_id} candidate {candidate_index}"
            )

        mime_type = str(image_part.get("mimeType", "image/png"))
        image_bytes = base64.b64decode(str(image_part["data"]))

        logger.info(
            "ref_pack_reference_request_completed",
            model=model,
            shot_id=shot.shot_id,
            candidate_index=candidate_index,
            image_bytes=len(image_bytes),
        )

        return GeneratedReferenceImage(
            image_bytes=image_bytes,
            mime_type=mime_type,
            provider_params={
                "response_modalities": ["TEXT", "IMAGE"],
                "provider": "google_generative_language_generate_content",
            },
        )


class RefPackGenerator:
    """Generate `ref_pack.v1` artifacts from compiled shot plans."""

    def __init__(
        self,
        *,
        image_generator: ReferenceImageGenerator | None = None,
        output_root: Path | None = None,
        now_factory: Callable[[], datetime] = _utc_now,
        reference_model: str = GeminiNanoBananaReferenceImageGenerator.DEFAULT_MODEL,
    ) -> None:
        self.reference_model = reference_model
        self.image_generator = image_generator or GeminiNanoBananaReferenceImageGenerator(
            model=reference_model
        )
        self.output_root = output_root or Path("storage/references")
        self.now_factory = now_factory

    async def generate(
        self,
        *,
        job_id: str,
        shot_plan: CompiledShotPlan | Mapping[str, Any] | None = None,
        review_payload: FirstFrameReviewPayload | Mapping[str, Any] | None = None,
        reference_assets: Sequence[FirstFrameReferenceAsset | Mapping[str, Any]] | None = None,
        aspect_ratio: str = "9:16",
        review_guidance: Sequence[str] | None = None,
        candidates_per_shot: int | None = None,
    ) -> RefPackV1:
        """Generate per-shot references and return the typed `ref_pack.v1` artifact."""
        if candidates_per_shot is not None and candidates_per_shot < 1:
            raise ValueError("candidates_per_shot must be >= 1")

        resolved_review_payload = _resolve_review_payload(
            shot_plan=shot_plan,
            review_payload=review_payload,
            reference_assets=reference_assets,
            aspect_ratio=aspect_ratio,
            review_guidance=review_guidance,
        )
        resolved_aspect_ratio = resolved_review_payload.aspect_ratio
        created_at = self.now_factory()

        shots: list[ShotReferenceGroup] = []
        for review_shot in resolved_review_payload.shots:
            candidate_count = candidates_per_shot or _resolve_candidate_count(review_shot)
            reference_candidates: list[ReferenceCandidate] = []

            for candidate_index in range(1, candidate_count + 1):
                prompt = _build_candidate_prompt(
                    review_shot,
                    candidate_index=candidate_index,
                    candidate_count=candidate_count,
                )
                generated = await self.image_generator.generate(
                    prompt,
                    candidate_index=candidate_index,
                    aspect_ratio=resolved_aspect_ratio,
                    model=self.reference_model,
                    shot=review_shot,
                    reference_assets=resolved_review_payload.reference_assets,
                )
                ref_id = _build_ref_id(review_shot.shot_id, candidate_index)
                asset_path = self._write_reference_asset(
                    job_id=job_id,
                    shot_id=review_shot.shot_id,
                    ref_id=ref_id,
                    image_bytes=generated.image_bytes,
                    mime_type=generated.mime_type,
                )
                reference_candidates.append(
                    ReferenceCandidate(
                        ref_id=ref_id,
                        asset_path=asset_path,
                        prompt_summary=_build_prompt_summary(review_shot, candidate_index),
                        model=self.reference_model,
                        params=ReferenceCandidateParams(
                            aspect_ratio=resolved_aspect_ratio,
                            candidate_index=candidate_index,
                            prompt=prompt,
                            first_frame_prompt_id=review_shot.first_frame_prompt_id,
                            reference_asset_ids=review_shot.reference_asset_ids,
                            provider_params=dict(generated.provider_params),
                        ),
                        created_at=self.now_factory(),
                    )
                )

            shots.append(
                ShotReferenceGroup(
                    shot_id=review_shot.shot_id,
                    reference_candidates=reference_candidates,
                )
            )

        return RefPackV1(
            ref_pack_id=_build_ref_pack_id(
                job_id=job_id,
                preset_id=resolved_review_payload.preset.preset_id,
                source_shot_plan_id=resolved_review_payload.source_plan_id,
                source_review_payload_id=resolved_review_payload.payload_id,
                created_at=created_at,
            ),
            job_id=job_id,
            preset_id=resolved_review_payload.preset.preset_id,
            source_shot_plan_id=resolved_review_payload.source_plan_id,
            lineage=_build_lineage(resolved_review_payload),
            shots=shots,
        )

    def _write_reference_asset(
        self,
        *,
        job_id: str,
        shot_id: str,
        ref_id: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        job_dir = self.output_root / _path_segment(job_id) / _path_segment(shot_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        extension = _extension_for_mime_type(mime_type)
        asset_path = job_dir / f"{ref_id}{extension}"
        asset_path.write_bytes(image_bytes)
        return str(asset_path)


def _extract_inline_image_part(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return None

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content")
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            inline_data = part.get("inlineData")
            if isinstance(inline_data, Mapping) and "data" in inline_data:
                return inline_data

    return None


def _resolve_review_payload(
    *,
    shot_plan: CompiledShotPlan | Mapping[str, Any] | None,
    review_payload: FirstFrameReviewPayload | Mapping[str, Any] | None,
    reference_assets: Sequence[FirstFrameReferenceAsset | Mapping[str, Any]] | None,
    aspect_ratio: str,
    review_guidance: Sequence[str] | None,
) -> FirstFrameReviewPayload:
    if (shot_plan is None) == (review_payload is None):
        raise ValueError("Provide exactly one of shot_plan or review_payload")

    if review_payload is not None:
        payload = (
            review_payload
            if isinstance(review_payload, FirstFrameReviewPayload)
            else FirstFrameReviewPayload.model_validate(review_payload)
        )
        if reference_assets is not None:
            raise ValueError("reference_assets cannot be overridden when review_payload is provided")
        if review_guidance is not None:
            raise ValueError("review_guidance cannot be overridden when review_payload is provided")
        if aspect_ratio != payload.aspect_ratio:
            raise ValueError(
                "aspect_ratio must match the supplied review_payload aspect_ratio"
            )
        return payload

    resolved_shot_plan = (
        shot_plan if isinstance(shot_plan, CompiledShotPlan) else CompiledShotPlan.model_validate(shot_plan)
    )
    return build_first_frame_review_payload(
        resolved_shot_plan,
        reference_assets=reference_assets,
        aspect_ratio=aspect_ratio,
        review_guidance=review_guidance,
    )


def _resolve_candidate_count(shot: FirstFrameReviewShot) -> int:
    counts = [requirement.count for requirement in shot.reference_requirements if requirement.count > 0]
    return max(counts) if counts else 1


def _build_candidate_prompt(
    shot: FirstFrameReviewShot,
    *,
    candidate_index: int,
    candidate_count: int,
) -> str:
    return "\n".join(
        [
            shot.review_prompt_text,
            (
                f"Generate candidate {candidate_index} of {candidate_count} for this shot. "
                "Keep the locked product and brand constraints intact."
            ),
            (
                "Vary the still direction slightly through composition, framing, depth, "
                "or lighting while preserving the same role and intent."
            ),
        ]
    )


def _build_prompt_summary(shot: FirstFrameReviewShot, candidate_index: int) -> str:
    return f"Shot {shot.sequence_order} {shot.role} candidate {candidate_index}: {shot.intent}"


def _build_ref_id(shot_id: str, candidate_index: int) -> str:
    return f"{shot_id}_ref_{candidate_index:02d}"


def _build_ref_pack_id(
    *,
    job_id: str,
    preset_id: str,
    source_shot_plan_id: str,
    source_review_payload_id: str,
    created_at: datetime,
) -> str:
    payload = {
        "job_id": job_id,
        "preset_id": preset_id,
        "source_shot_plan_id": source_shot_plan_id,
        "source_review_payload_id": source_review_payload_id,
        "created_at": created_at.isoformat(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"refpack_{_slug(preset_id)}_{digest}"


def _build_lineage(review_payload: FirstFrameReviewPayload) -> RefPackLineage:
    return RefPackLineage(
        preset_id=review_payload.preset.preset_id,
        preset_version=review_payload.preset.version,
        source_plan_id=review_payload.source_plan_id,
        source_review_payload_id=review_payload.payload_id,
        aspect_ratio=review_payload.aspect_ratio,
        reference_asset_ids=[asset.asset_id for asset in review_payload.reference_assets],
        review_guidance=list(review_payload.review_guidance),
    )


def _path_segment(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    return cleaned or "item"


def _extension_for_mime_type(mime_type: str) -> str:
    mime_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    return mime_map.get(mime_type.lower(), ".bin")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"
