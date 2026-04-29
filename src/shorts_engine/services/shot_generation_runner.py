"""Run typed shot take requests through the existing video-generation seam."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shorts_engine.adapters.video_gen.base import VideoGenProvider, VideoGenRequest, VideoGenResult
from shorts_engine.contracts.shot_take_v1 import (
    SHOT_TAKE_SCHEMA_VERSION,
    ShotTakeArtifact,
    ShotTakeBatchV1,
    ShotTakeLineage,
    ShotTakeParams,
    ShotTakeStatus,
)
from shorts_engine.logging import get_logger
from shorts_engine.shot_plans.contracts import ShotTakeRequest, TakeRequestStatus

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ShotGenerationRunner:
    """Narrow runner that turns one typed take request into typed take artifacts."""

    def __init__(
        self,
        *,
        video_provider: VideoGenProvider | None = None,
        output_root: Path | None = None,
        now_factory: Callable[[], datetime] = _utc_now,
        default_model: str | None = None,
    ) -> None:
        self.video_provider = video_provider or _default_video_provider()
        self.output_root = output_root or Path("storage/shot_takes")
        self.now_factory = now_factory
        self.default_model = default_model

    async def generate(
        self,
        *,
        job_id: str,
        take_request: ShotTakeRequest | Mapping[str, Any],
        reference_asset_paths: Sequence[str | Path] | None = None,
        take_count: int | None = None,
        aspect_ratio: str = "9:16",
        provider_params: Mapping[str, Any] | None = None,
    ) -> ShotTakeBatchV1:
        """Generate one batch of takes for a typed shot request."""
        request = (
            take_request
            if isinstance(take_request, ShotTakeRequest)
            else ShotTakeRequest.model_validate(take_request)
        )
        resolved_reference_paths = _merge_reference_asset_paths(request, reference_asset_paths)
        resolved_take_count = take_count or request.generation_defaults.target_take_count
        if resolved_take_count < 1:
            raise ValueError("take_count must be >= 1")

        provider_options = dict(provider_params or {})
        blocking_error = _blocking_error_message(
            request,
            resolved_reference_paths,
            video_provider=self.video_provider,
        )
        if blocking_error:
            logger.warning(
                "shot_take_request_blocked",
                job_id=job_id,
                shot_id=request.shot_id,
                take_request_id=request.take_request_id,
                reason=blocking_error,
            )
            return self._build_failed_batch(
                job_id=job_id,
                take_request=request,
                take_count=resolved_take_count,
                aspect_ratio=aspect_ratio,
                reference_asset_paths=resolved_reference_paths,
                provider_params=provider_options,
                error_message=blocking_error,
            )

        try:
            reference_images = _load_reference_images(resolved_reference_paths)
        except Exception as exc:
            logger.warning(
                "shot_take_reference_load_failed",
                job_id=job_id,
                shot_id=request.shot_id,
                take_request_id=request.take_request_id,
                error=str(exc),
            )
            return self._build_failed_batch(
                job_id=job_id,
                take_request=request,
                take_count=resolved_take_count,
                aspect_ratio=aspect_ratio,
                reference_asset_paths=resolved_reference_paths,
                provider_params=provider_options,
                error_message=str(exc),
            )

        takes: list[ShotTakeArtifact] = []
        batch_created_at = self.now_factory()
        for take_index in range(1, resolved_take_count + 1):
            seed = _resolve_seed(job_id, request, take_index)
            request_provider_params = dict(provider_options)
            if seed is not None:
                request_provider_params.setdefault("seed", seed)
            artifact_provider_params = _sanitize_provider_params_for_contract(
                request_provider_params
            )

            take_params = ShotTakeParams(
                aspect_ratio=aspect_ratio,
                duration_seconds=request.duration_target_seconds,
                prompt=_build_take_prompt(
                    request,
                    take_index=take_index,
                    take_count=resolved_take_count,
                ),
                negative_prompt=_build_negative_prompt(request),
                take_index=take_index,
                reference_asset_paths=resolved_reference_paths,
                variation_hint=_variation_hint_for_take(request, take_index),
                provider_params=artifact_provider_params,
            )

            take_id = _build_take_id(request.take_request_id, take_index)
            created_at = self.now_factory()
            try:
                result = await self.video_provider.generate(
                    VideoGenRequest(
                        prompt=take_params.prompt,
                        duration_seconds=max(1, round(request.duration_target_seconds)),
                        aspect_ratio=aspect_ratio,
                        negative_prompt=take_params.negative_prompt,
                        reference_images=reference_images or None,
                        options=request_provider_params or None,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "shot_take_generation_failed",
                    job_id=job_id,
                    shot_id=request.shot_id,
                    take_request_id=request.take_request_id,
                    take_id=take_id,
                    error=str(exc),
                )
                result = VideoGenResult(success=False, error_message=str(exc))

            takes.append(
                self._build_take_artifact(
                    job_id=job_id,
                    take_request=request,
                    take_id=take_id,
                    take_params=take_params,
                    take_index=take_index,
                    seed=seed,
                    result=result,
                    created_at=created_at,
                    reference_asset_paths=resolved_reference_paths,
                )
            )

        return ShotTakeBatchV1(
            take_batch_id=_build_take_batch_id(
                job_id=job_id,
                take_request_id=request.take_request_id,
                created_at=batch_created_at,
            ),
            schema_version=SHOT_TAKE_SCHEMA_VERSION,
            job_id=job_id,
            shot_id=request.shot_id,
            take_request_id=request.take_request_id,
            takes=takes,
        )

    def _build_failed_batch(
        self,
        *,
        job_id: str,
        take_request: ShotTakeRequest,
        take_count: int,
        aspect_ratio: str,
        reference_asset_paths: Sequence[str],
        provider_params: Mapping[str, Any],
        error_message: str,
    ) -> ShotTakeBatchV1:
        created_at = self.now_factory()
        takes = [
            ShotTakeArtifact(
                take_id=_build_take_id(take_request.take_request_id, take_index),
                job_id=job_id,
                shot_id=take_request.shot_id,
                take_request_id=take_request.take_request_id,
                asset_path=None,
                model=_resolve_model_name(
                    self.video_provider,
                    metadata=None,
                    default_model=self.default_model,
                ),
                params=ShotTakeParams(
                    aspect_ratio=aspect_ratio,
                    duration_seconds=take_request.duration_target_seconds,
                    prompt=_build_take_prompt(
                        take_request,
                        take_index=take_index,
                        take_count=take_count,
                    ),
                    negative_prompt=_build_negative_prompt(take_request),
                    take_index=take_index,
                    reference_asset_paths=list(reference_asset_paths),
                    variation_hint=_variation_hint_for_take(take_request, take_index),
                    provider_params=_sanitize_provider_params_for_contract(provider_params),
                ),
                seed=_resolve_seed(job_id, take_request, take_index),
                cost_estimate=None,
                created_at=created_at,
                status=ShotTakeStatus.FAILED,
                error_message=error_message,
                lineage=_build_lineage(
                    take_request,
                    reference_asset_paths=reference_asset_paths,
                    provider_job_id=None,
                ),
                provider_metadata={},
            )
            for take_index in range(1, take_count + 1)
        ]

        return ShotTakeBatchV1(
            take_batch_id=_build_take_batch_id(
                job_id=job_id,
                take_request_id=take_request.take_request_id,
                created_at=created_at,
            ),
            schema_version=SHOT_TAKE_SCHEMA_VERSION,
            job_id=job_id,
            shot_id=take_request.shot_id,
            take_request_id=take_request.take_request_id,
            takes=takes,
        )

    def _build_take_artifact(
        self,
        *,
        job_id: str,
        take_request: ShotTakeRequest,
        take_id: str,
        take_params: ShotTakeParams,
        take_index: int,
        seed: int | None,
        result: VideoGenResult,
        created_at: datetime,
        reference_asset_paths: Sequence[str],
    ) -> ShotTakeArtifact:
        metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
        provider_job_id = _extract_provider_job_id(metadata)
        model_name = _resolve_model_name(
            self.video_provider,
            metadata=metadata,
            default_model=self.default_model,
        )

        if not result.success:
            return ShotTakeArtifact(
                take_id=take_id,
                job_id=job_id,
                shot_id=take_request.shot_id,
                take_request_id=take_request.take_request_id,
                asset_path=None,
                model=model_name,
                params=take_params,
                seed=seed,
                cost_estimate=_extract_cost_estimate(metadata),
                created_at=created_at,
                status=ShotTakeStatus.FAILED,
                error_message=result.error_message or "Video generation failed",
                lineage=_build_lineage(
                    take_request,
                    reference_asset_paths=reference_asset_paths,
                    provider_job_id=provider_job_id,
                ),
                provider_metadata=dict(metadata),
            )

        asset_path = _resolve_asset_path(
            output_root=self.output_root,
            job_id=job_id,
            shot_id=take_request.shot_id,
            take_id=take_id,
            result=result,
        )
        if not asset_path:
            return ShotTakeArtifact(
                take_id=take_id,
                job_id=job_id,
                shot_id=take_request.shot_id,
                take_request_id=take_request.take_request_id,
                asset_path=None,
                model=model_name,
                params=take_params,
                seed=seed,
                cost_estimate=_extract_cost_estimate(metadata),
                created_at=created_at,
                status=ShotTakeStatus.FAILED,
                error_message="Generation completed without a video asset",
                lineage=_build_lineage(
                    take_request,
                    reference_asset_paths=reference_asset_paths,
                    provider_job_id=provider_job_id,
                ),
                provider_metadata=dict(metadata),
            )

        logger.info(
            "shot_take_generated",
            job_id=job_id,
            shot_id=take_request.shot_id,
            take_request_id=take_request.take_request_id,
            take_id=take_id,
            take_index=take_index,
            provider=self.video_provider.name,
            provider_job_id=provider_job_id,
            asset_path=asset_path,
        )

        return ShotTakeArtifact(
            take_id=take_id,
            job_id=job_id,
            shot_id=take_request.shot_id,
            take_request_id=take_request.take_request_id,
            asset_path=asset_path,
            model=model_name,
            params=take_params,
            seed=seed,
            cost_estimate=_extract_cost_estimate(metadata),
            created_at=created_at,
            status=ShotTakeStatus.GENERATED,
            lineage=_build_lineage(
                take_request,
                reference_asset_paths=reference_asset_paths,
                provider_job_id=provider_job_id,
            ),
            provider_metadata=dict(metadata),
        )


def _default_video_provider() -> VideoGenProvider:
    from shorts_engine.jobs.video_pipeline import get_video_gen_provider

    return get_video_gen_provider()


def _blocking_error_message(
    take_request: ShotTakeRequest,
    reference_asset_paths: Sequence[str],
    *,
    video_provider: VideoGenProvider,
) -> str | None:
    if take_request.status == TakeRequestStatus.BLOCKED_ON_REFERENCES:
        return "Take request is blocked on approved references"

    if take_request.status != TakeRequestStatus.READY:
        return f"Take request must be READY before generation (got {take_request.status})"

    if (
        take_request.generation_defaults.requires_approved_reference
        and take_request.approved_board is None
    ):
        return "Approved storyboard board is required before take generation"

    if take_request.generation_defaults.requires_approved_reference and not reference_asset_paths:
        return "Approved storyboard board asset is required before take generation"

    if (
        take_request.generation_defaults.requires_approved_reference
        and reference_asset_paths
        and not video_provider.supports_reference_images
    ):
        return (
            "Configured video provider does not support approved-board reference inputs "
            "required for shot-plan generation"
        )

    return None


def _load_reference_images(reference_asset_paths: Sequence[str]) -> list[bytes]:
    images: list[bytes] = []
    for value in reference_asset_paths:
        if _looks_like_url(value):
            continue
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Reference asset not found: {value}")
        images.append(path.read_bytes())
    return images


def _build_take_prompt(
    take_request: ShotTakeRequest,
    *,
    take_index: int,
    take_count: int,
) -> str:
    variation_hint = _variation_hint_for_take(take_request, take_index)
    parts = [
        take_request.camera_language,
        take_request.subject,
        take_request.environment,
        take_request.motion_beat,
        f"Storyboard visual world: {take_request.storyboard_deck.visual_world}",
        f"Storyboard layout system: {take_request.storyboard_deck.layout_system}",
        f"Storyboard layout notes: {take_request.storyboard_board.layout_notes}",
        "Sequence continuity locks: "
        + " ".join(take_request.storyboard_deck.continuity_locks),
        f"Intent: {take_request.intent}",
        (
            f"Generate take {take_index} of {take_count} for this shot while preserving the "
            "approved storyboard board / first-frame direction."
        ),
    ]
    if take_request.storyboard_board.title:
        parts.append(f"Board title: {take_request.storyboard_board.title}")
    if take_request.storyboard_board.hook_role:
        parts.append(f"Board hook role: {take_request.storyboard_board.hook_role}")
    if (
        take_request.generation_defaults.preserve_approved_board_text
        and take_request.storyboard_board.on_frame_text
    ):
        parts.append(
            "Preserve this approved on-frame board copy exactly as the designed motion source: "
            + take_request.storyboard_board.on_frame_text
        )
    if variation_hint:
        parts.append(f"Variation hint: {variation_hint}")
    if take_request.generation_defaults.notes:
        parts.append("Generation notes: " + " ".join(take_request.generation_defaults.notes))
    if (
        take_request.generation_defaults.avoid_visible_text
        and not take_request.generation_defaults.preserve_approved_board_text
    ):
        parts.append("Avoid readable text, letters, numbers, labels, captions, and subtitles.")
    return ". ".join(_clean_sentence(part) for part in parts if part)


def _build_negative_prompt(take_request: ShotTakeRequest) -> str | None:
    if (
        take_request.generation_defaults.preserve_approved_board_text
        and take_request.storyboard_board.on_frame_text
    ):
        return "extra readable text, changed board copy, fake labels, subtitles, UI overlays"
    if not take_request.generation_defaults.avoid_visible_text:
        return None
    return "readable text, letters, numbers, labels, captions, subtitles, UI overlays"


def _variation_hint_for_take(take_request: ShotTakeRequest, take_index: int) -> str | None:
    if take_index <= len(take_request.variation_hints):
        return take_request.variation_hints[take_index - 1]
    return None


def _resolve_seed(job_id: str, take_request: ShotTakeRequest, take_index: int) -> int | None:
    if take_request.generation_defaults.seed_policy != "deterministic_per_shot_take":
        return None

    payload = {
        "job_id": job_id,
        "shot_id": take_request.shot_id,
        "take_request_id": take_request.take_request_id,
        "take_index": take_index,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _sanitize_provider_params_for_contract(provider_params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider params JSON-safe in typed take artifacts."""
    return {
        str(key): _sanitize_provider_param_value(value)
        for key, value in provider_params.items()
    }


def _sanitize_provider_param_value(value: Any) -> Any:
    """Replace raw binary payloads with compact metadata summaries."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary:{len(bytes(value))} bytes>"
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_provider_param_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_provider_param_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_provider_param_value(item) for item in value]
    return value


def _build_take_id(take_request_id: str, take_index: int) -> str:
    return f"{take_request_id}_take_{take_index:02d}"


def _build_take_batch_id(
    *,
    job_id: str,
    take_request_id: str,
    created_at: datetime,
) -> str:
    payload = {
        "job_id": job_id,
        "take_request_id": take_request_id,
        "created_at": created_at.isoformat(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"takebatch_{_slug(take_request_id)}_{digest}"


def _resolve_model_name(
    provider: VideoGenProvider,
    *,
    metadata: Mapping[str, Any] | None,
    default_model: str | None,
) -> str:
    if metadata:
        model = metadata.get("model")
        if isinstance(model, str) and model.strip():
            return model

    if default_model and default_model.strip():
        return default_model

    provider_model = getattr(provider, "model", None)
    if isinstance(provider_model, str) and provider_model.strip():
        return provider_model

    provider_name = getattr(provider, "name", None)
    if isinstance(provider_name, str) and provider_name.strip():
        return provider_name

    return provider.__class__.__name__


def _resolve_asset_path(
    *,
    output_root: Path,
    job_id: str,
    shot_id: str,
    take_id: str,
    result: VideoGenResult,
) -> str | None:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if result.video_data:
        asset_dir = output_root / _path_segment(job_id) / _path_segment(shot_id)
        asset_dir.mkdir(parents=True, exist_ok=True)
        extension = _extension_for_video_asset(metadata)
        asset_path = asset_dir / f"{take_id}{extension}"
        asset_path.write_bytes(result.video_data)
        return str(asset_path)

    video_url = metadata.get("video_url")
    if isinstance(video_url, str) and video_url.strip():
        return video_url

    return None


def _extract_provider_job_id(metadata: Mapping[str, Any]) -> str | None:
    for key in ("generation_id", "operation_name", "job_id", "request_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_cost_estimate(metadata: Mapping[str, Any]) -> float | None:
    for key in ("cost_estimate", "estimated_cost_usd", "cost_usd"):
        value = metadata.get(key)
        if isinstance(value, int | float) and value >= 0:
            return float(value)
    return None


def _build_lineage(
    take_request: ShotTakeRequest,
    *,
    reference_asset_paths: Sequence[str],
    provider_job_id: str | None,
) -> ShotTakeLineage:
    source_plan_id = None
    for key in ("source_plan_id", "plan_id"):
        value = take_request.metadata.get(key)
        if isinstance(value, str) and value.strip():
            source_plan_id = value
            break

    return ShotTakeLineage(
        preset_id=take_request.preset_id,
        preset_version=take_request.preset_version,
        sequence_order=take_request.sequence_order,
        source_plan_id=source_plan_id,
        approved_board_ref_id=(
            take_request.approved_board.ref_id if take_request.approved_board else None
        ),
        approved_board_asset_path=(
            take_request.approved_board.asset_path if take_request.approved_board else None
        ),
        approved_board_source_ref_pack_id=(
            take_request.approved_board.source_ref_pack_id if take_request.approved_board else None
        ),
        approved_board_source_review_payload_id=(
            take_request.approved_board.source_review_payload_id
            if take_request.approved_board
            else None
        ),
        reference_asset_paths=list(reference_asset_paths),
        provider_job_id=provider_job_id,
        request_metadata=dict(take_request.metadata),
    )


def _extension_for_video_asset(metadata: Mapping[str, Any]) -> str:
    video_url = metadata.get("video_url")
    if isinstance(video_url, str) and "." in video_url.rsplit("/", 1)[-1]:
        suffix = Path(video_url.split("?", 1)[0]).suffix
        if suffix:
            return suffix
    return ".mp4"


def _path_segment(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    return cleaned or "item"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def _clean_sentence(value: str) -> str:
    return value.strip().rstrip(".")


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _merge_reference_asset_paths(
    take_request: ShotTakeRequest,
    reference_asset_paths: Sequence[str | Path] | None,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    approved_board_path = (
        take_request.approved_board.asset_path if take_request.approved_board else None
    )
    if approved_board_path:
        normalized = str(approved_board_path)
        merged.append(normalized)
        seen.add(normalized)

    for value in reference_asset_paths or []:
        normalized = str(value)
        if normalized in seen:
            continue
        merged.append(normalized)
        seen.add(normalized)

    return merged
