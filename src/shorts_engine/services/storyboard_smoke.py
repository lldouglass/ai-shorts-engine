"""Deterministic storyboard-first smoke runner for local end-to-end proof."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import textwrap
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from shorts_engine.adapters.video_gen.base import VideoGenProvider, VideoGenRequest, VideoGenResult
from shorts_engine.contracts.shot_take_v1 import ShotTakeBatchV1
from shorts_engine.services.ref_pack_generator import GeneratedReferenceImage, RefPackGenerator
from shorts_engine.services.shot_generation_runner import ShotGenerationRunner
from shorts_engine.shot_plans import apply_ref_pack_approvals
from shorts_engine.shot_plans.benchmarks import (
    TALL_OWL_BENCHMARK_ID,
    build_tall_owl_first_frame_review_payload,
    compile_tall_owl_benchmark_shot_plan,
)
from shorts_engine.shot_plans.contracts import CompiledShotPlan, FirstFrameReviewPayload

DEFAULT_STORYBOARD_SMOKE_JOB_ID = "storyboard_smoke_tall_owl_v1"
DEFAULT_STORYBOARD_SMOKE_OUTPUT_ROOT = Path("output/storyboard_smoke")
DEFAULT_CANDIDATES_PER_SHOT = 2
DEFAULT_TAKE_COUNT = 1
SMOKE_REFERENCE_MODEL = "storyboard-smoke-reference-v1"
SMOKE_MOTION_MODEL = "storyboard-smoke-motion-v1"
SMOKE_CREATED_AT = datetime(2026, 4, 23, 18, 10, tzinfo=UTC)


@dataclass(frozen=True)
class StoryboardSmokeArtifactPaths:
    """Filesystem outputs for one storyboard smoke run."""

    run_dir: str
    compiled_shot_plan_path: str
    first_frame_review_payload_path: str
    ref_pack_path: str
    approvals_path: str
    approved_shot_plan_path: str
    shot_take_batch_paths: list[str]
    summary_path: str


@dataclass(frozen=True)
class StoryboardSmokeResult:
    """Summary returned from the storyboard-first smoke runner."""

    benchmark_id: str
    job_id: str
    created_at: str
    candidates_per_shot: int
    take_count: int
    selection_strategy: str
    plan_id: str
    review_payload_id: str
    ref_pack_id: str
    approved_ref_ids_by_shot: dict[str, str]
    shot_take_batch_ids: list[str]
    reference_candidate_asset_paths: list[str]
    motion_take_asset_paths: list[str]
    artifacts: StoryboardSmokeArtifactPaths

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return asdict(self)


class StoryboardSmokeReferenceImageGenerator:
    """Deterministic image generator that produces inspectable storyboard boards."""

    def __init__(self) -> None:
        self.canvas_size = (720, 1280)

    async def generate(
        self,
        prompt: str,
        *,
        candidate_index: int,
        aspect_ratio: str,
        model: str,
        shot: Any,
        reference_assets: Any,
    ) -> GeneratedReferenceImage:
        del aspect_ratio, model, reference_assets

        image = Image.new(
            "RGB",
            self.canvas_size,
            color=_background_color(shot.shot_id, candidate_index),
        )
        draw = ImageDraw.Draw(image)
        copy_box = (56, 72, 664, 220)
        hero_box = (72, 290, 648, 1032)
        footer_box = (56, 1084, 664, 1220)

        draw.rounded_rectangle(copy_box, radius=24, fill=(244, 239, 227))
        draw.rounded_rectangle(
            hero_box,
            radius=36,
            fill=(255, 250, 242),
            outline=(43, 65, 55),
            width=4,
        )
        draw.rounded_rectangle(footer_box, radius=20, fill=(234, 227, 213))

        draw.rectangle(
            (156, 386, 564, 910),
            fill=(232, 189, 118),
            outline=(61, 54, 42),
            width=5,
        )
        draw.ellipse(
            (264, 530, 456, 722),
            fill=(250, 245, 236),
            outline=(91, 76, 62),
            width=4,
        )
        draw.rectangle((214, 928, 506, 966), fill=(38, 52, 46))

        title = shot.storyboard_board.title or shot.role.replace("_", " ").title()
        board_copy = shot.storyboard_board.on_frame_text or shot.intent
        footer = f"{shot.sequence_order}. {shot.role} | candidate {candidate_index}"
        prompt_excerpt = _truncate(
            textwrap.shorten(prompt, width=120, placeholder="..."),
            limit=120,
        )

        _draw_wrapped_text(
            draw,
            (88, 104),
            title,
            line_height=24,
            width=40,
            fill=(26, 32, 25),
        )
        _draw_wrapped_text(
            draw,
            (88, 142),
            board_copy,
            line_height=22,
            width=42,
            fill=(26, 32, 25),
        )
        _draw_wrapped_text(
            draw,
            (100, 1098),
            footer,
            line_height=22,
            width=52,
            fill=(33, 39, 31),
        )
        _draw_wrapped_text(
            draw,
            (100, 1132),
            prompt_excerpt,
            line_height=18,
            width=64,
            fill=(69, 77, 62),
        )

        image_bytes = _image_bytes(image, format="PNG")
        return GeneratedReferenceImage(
            image_bytes=image_bytes,
            mime_type="image/png",
            provider_params={
                "provider": "storyboard_smoke_reference_generator",
                "candidate_index": candidate_index,
            },
        )


class StoryboardSmokeVideoProvider(VideoGenProvider):
    """Deterministic motion stub that turns approved boards into inspectable GIF takes."""

    def __init__(self, *, model: str = SMOKE_MOTION_MODEL) -> None:
        self.model = model

    @property
    def name(self) -> str:
        return "storyboard-smoke"

    @property
    def supports_reference_images(self) -> bool:
        return True

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        base_image = _base_motion_image(request)
        frames = _build_motion_frames(base_image, request.prompt, request.options or {})
        video_data = _gif_bytes(frames)
        request_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:12]

        return VideoGenResult(
            success=True,
            video_data=video_data,
            duration_seconds=float(request.duration_seconds),
            metadata={
                "generation_id": f"storyboard-smoke-{request_hash}",
                "model": self.model,
                "provider": self.name,
                "cost_estimate": 0.0,
                "video_url": f"https://storyboard-smoke.local/{request_hash}.gif",
            },
        )

    async def check_status(self, job_id: str) -> dict[str, Any]:
        return {"job_id": job_id, "status": "completed", "progress": 100}


async def run_storyboard_first_smoke(
    *,
    benchmark_id: str = TALL_OWL_BENCHMARK_ID,
    output_root: Path = DEFAULT_STORYBOARD_SMOKE_OUTPUT_ROOT,
    job_id: str = DEFAULT_STORYBOARD_SMOKE_JOB_ID,
    candidates_per_shot: int = DEFAULT_CANDIDATES_PER_SHOT,
    take_count: int = DEFAULT_TAKE_COUNT,
) -> StoryboardSmokeResult:
    """Run the smallest real storyboard-first flow end to end and write artifacts."""
    if candidates_per_shot < 1:
        raise ValueError("candidates_per_shot must be >= 1")
    if take_count < 1:
        raise ValueError("take_count must be >= 1")

    plan, review_payload = _resolve_benchmark(benchmark_id)
    run_root = output_root.expanduser().resolve() / _path_segment(job_id)
    run_root.mkdir(parents=True, exist_ok=True)

    compiled_path = _write_json(
        run_root / "compiled_shot_plan.json",
        plan.model_dump(mode="json"),
    )
    review_payload_path = _write_json(
        run_root / "first_frame_review_payload.json",
        review_payload.model_dump(mode="json"),
    )

    ref_pack_generator = RefPackGenerator(
        image_generator=StoryboardSmokeReferenceImageGenerator(),
        output_root=run_root / "reference_candidates",
        now_factory=lambda: SMOKE_CREATED_AT,
        reference_model=SMOKE_REFERENCE_MODEL,
    )
    ref_pack = await ref_pack_generator.generate(
        job_id=job_id,
        review_payload=review_payload,
        candidates_per_shot=candidates_per_shot,
    )
    ref_pack_path = _write_json(run_root / "ref_pack.v1.json", ref_pack.model_dump(mode="json"))

    approved_ref_ids_by_shot = {
        shot_group.shot_id: shot_group.reference_candidates[0].ref_id
        for shot_group in ref_pack.shots
    }
    approvals_path = _write_json(
        run_root / "approved_ref_ids_by_shot.json",
        {
            "selection_strategy": "first_candidate_per_shot",
            "approved_ref_ids_by_shot": approved_ref_ids_by_shot,
        },
    )

    approved_plan = apply_ref_pack_approvals(
        plan,
        ref_pack,
        approved_ref_ids_by_shot,
    )
    approved_shot_plan_path = _write_json(
        run_root / "approved_shot_plan.json",
        approved_plan.model_dump(mode="json"),
    )

    take_batches_dir = run_root / "shot_take_batches"
    take_batches_dir.mkdir(parents=True, exist_ok=True)
    shot_take_runner = ShotGenerationRunner(
        video_provider=StoryboardSmokeVideoProvider(),
        output_root=run_root / "motion_takes",
        now_factory=lambda: SMOKE_CREATED_AT,
        default_model=SMOKE_MOTION_MODEL,
    )

    shot_take_batches: list[ShotTakeBatchV1] = []
    shot_take_batch_paths: list[str] = []
    motion_take_asset_paths: list[str] = []

    ref_groups_by_shot = {group.shot_id: group for group in ref_pack.shots}
    for shot in approved_plan.shots:
        ref_group = ref_groups_by_shot[shot.shot_id]
        batch = await shot_take_runner.generate(
            job_id=job_id,
            take_request=shot.take_request,
            reference_asset_paths=[
                candidate.asset_path for candidate in ref_group.reference_candidates
            ],
            take_count=take_count,
        )
        shot_take_batches.append(batch)
        shot_take_batch_paths.append(
            _write_json(
                take_batches_dir / f"{shot.shot_id}.shot_take.v1.json",
                batch.model_dump(mode="json"),
            )
        )
        motion_take_asset_paths.extend(
            str(Path(asset_path).resolve())
            for asset_path in [take.asset_path for take in batch.takes]
            if asset_path
        )

    summary = StoryboardSmokeResult(
        benchmark_id=TALL_OWL_BENCHMARK_ID,
        job_id=job_id,
        created_at=SMOKE_CREATED_AT.isoformat(),
        candidates_per_shot=candidates_per_shot,
        take_count=take_count,
        selection_strategy="first_candidate_per_shot",
        plan_id=plan.plan_id,
        review_payload_id=review_payload.payload_id,
        ref_pack_id=ref_pack.ref_pack_id,
        approved_ref_ids_by_shot=approved_ref_ids_by_shot,
        shot_take_batch_ids=[batch.take_batch_id for batch in shot_take_batches],
        reference_candidate_asset_paths=[
            str(Path(candidate.asset_path).resolve())
            for shot_group in ref_pack.shots
            for candidate in shot_group.reference_candidates
        ],
        motion_take_asset_paths=motion_take_asset_paths,
        artifacts=StoryboardSmokeArtifactPaths(
            run_dir=str(run_root.resolve()),
            compiled_shot_plan_path=compiled_path,
            first_frame_review_payload_path=review_payload_path,
            ref_pack_path=ref_pack_path,
            approvals_path=approvals_path,
            approved_shot_plan_path=approved_shot_plan_path,
            shot_take_batch_paths=shot_take_batch_paths,
            summary_path=str((run_root / "storyboard_first_smoke_summary.json").resolve()),
        ),
    )
    _write_json(Path(summary.artifacts.summary_path), summary.to_dict())
    return summary


def run_storyboard_first_smoke_sync(
    *,
    benchmark_id: str = TALL_OWL_BENCHMARK_ID,
    output_root: Path = DEFAULT_STORYBOARD_SMOKE_OUTPUT_ROOT,
    job_id: str = DEFAULT_STORYBOARD_SMOKE_JOB_ID,
    candidates_per_shot: int = DEFAULT_CANDIDATES_PER_SHOT,
    take_count: int = DEFAULT_TAKE_COUNT,
) -> StoryboardSmokeResult:
    """Synchronous wrapper for CLI use."""
    return asyncio.run(
        run_storyboard_first_smoke(
            benchmark_id=benchmark_id,
            output_root=output_root,
            job_id=job_id,
            candidates_per_shot=candidates_per_shot,
            take_count=take_count,
        )
    )


def _resolve_benchmark(
    benchmark_id: str,
) -> tuple[CompiledShotPlan, FirstFrameReviewPayload]:
    normalized = benchmark_id.strip().lower()
    if normalized in {"tall-owl", TALL_OWL_BENCHMARK_ID}:
        return (
            compile_tall_owl_benchmark_shot_plan(),
            build_tall_owl_first_frame_review_payload(),
        )

    raise ValueError(f"Unknown storyboard smoke benchmark: {benchmark_id}")


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(path.resolve())


def _path_segment(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    return cleaned or "item"


def _background_color(shot_id: str, candidate_index: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{shot_id}:{candidate_index}".encode()).digest()
    return (
        28 + digest[0] % 40,
        44 + digest[1] % 40,
        48 + digest[2] % 40,
    )


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    width: int,
    line_height: int,
    fill: tuple[int, int, int],
) -> None:
    x, y = position
    for line in textwrap.wrap(text, width=width):
        draw.text((x, y), line, fill=fill)
        y += line_height


def _image_bytes(image: Image.Image, *, format: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def _base_motion_image(request: VideoGenRequest) -> Image.Image:
    if request.reference_images:
        return Image.open(io.BytesIO(request.reference_images[0])).convert("RGB")

    image = Image.new("RGB", (720, 1280), color=(38, 58, 54))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((72, 120, 648, 1120), radius=36, fill=(244, 239, 227))
    _draw_wrapped_text(
        draw,
        (112, 168),
        "Storyboard smoke fallback board",
        width=30,
        line_height=24,
        fill=(33, 39, 31),
    )
    return image


def _build_motion_frames(
    base_image: Image.Image,
    prompt: str,
    options: dict[str, Any],
) -> list[Image.Image]:
    canvas = ImageOps.contain(base_image, (432, 768)).convert("RGB")
    frames: list[Image.Image] = []
    seed_value = str(options.get("seed", "seedless"))
    for index in range(3):
        frame = canvas.copy()
        draw = ImageDraw.Draw(frame)
        accent = _motion_accent_color(seed_value, index)
        inset = 18 + index * 10
        draw.rounded_rectangle(
            (inset, inset, frame.width - inset, frame.height - inset),
            radius=28,
            outline=accent,
            width=8,
        )
        draw.rectangle((36, frame.height - 104, frame.width - 36, frame.height - 68), fill=accent)
        _draw_wrapped_text(
            draw,
            (44, frame.height - 96),
            _truncate(textwrap.shorten(prompt, width=56, placeholder="..."), limit=56),
            width=44,
            line_height=18,
            fill=(248, 244, 236),
        )
        frames.append(frame)
    return frames


def _motion_accent_color(seed_value: str, frame_index: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{seed_value}:{frame_index}".encode()).digest()
    return (
        132 + digest[0] % 90,
        92 + digest[1] % 80,
        44 + digest[2] % 60,
    )


def _gif_bytes(frames: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
    )
    return buffer.getvalue()


def _truncate(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 1)].rstrip() + "..."
