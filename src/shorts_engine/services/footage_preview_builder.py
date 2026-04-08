"""Build narrow preview artifacts for uploaded footage without rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "uploaded_footage_preview_bundle.v1"
TRANSCRIPT_SCHEMA_VERSION = "uploaded_footage_transcript.v1"
CUT_PLAN_SCHEMA_VERSION = "uploaded_footage_cut_plan.v1"
CAPTION_PLAN_SCHEMA_VERSION = "uploaded_footage_caption_plan.v1"
PACKET_SCHEMA_VERSION = "shortform_preview_packet.v1"
MIN_CLIP_DURATION_SECONDS = 0.5


@dataclass(frozen=True)
class UploadedFootageTranscriptSegment:
    """Time-aligned transcript segment derived from uploaded footage."""

    segment_id: str
    start_seconds: float
    end_seconds: float
    text: str
    speaker_label: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class UploadedFootageSource:
    """Minimal source reference for uploaded Logan footage previewing."""

    source_id: str
    source_uri: str
    file_name: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    transcript_segments: tuple[UploadedFootageTranscriptSegment, ...] = ()


@dataclass(frozen=True)
class TranscriptArtifact:
    """Normalized transcript artifact with stable segment ordering."""

    schema_version: str
    source_id: str
    full_text: str
    language: str
    segments: tuple[UploadedFootageTranscriptSegment, ...]


@dataclass(frozen=True)
class CutPlanStep:
    """One source-range selection in the preview cut plan."""

    clip_id: str
    source_id: str
    start_seconds: float
    end_seconds: float
    transcript_segment_ids: tuple[str, ...]
    intent: Literal["hook", "body", "cta", "bridge"]


@dataclass(frozen=True)
class CutPlanArtifact:
    """Ordered cut plan for a lightweight review preview."""

    schema_version: str
    source_id: str
    total_duration_seconds: float
    steps: tuple[CutPlanStep, ...]


@dataclass(frozen=True)
class CaptionPlanCue:
    """Caption cue aligned to preview clips."""

    cue_id: str
    start_seconds: float
    end_seconds: float
    text: str
    transcript_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class CaptionPlanArtifact:
    """Stable caption plan for later packet compilation."""

    schema_version: str
    source_id: str
    style_preset: str
    cues: tuple[CaptionPlanCue, ...]


@dataclass(frozen=True)
class PreviewPacketScene:
    """Packet-friendly scene entry aligned to one cut-plan step."""

    scene_id: str
    scene_number: int
    source_id: str
    start_seconds: float
    end_seconds: float
    transcript_segment_ids: tuple[str, ...]
    caption_cue_ids: tuple[str, ...]
    primary_text: str
    asset_role: Literal["uploaded_footage"]


@dataclass(frozen=True)
class PreviewPacketAsset:
    """Asset reference for later TypeScript packet compilation."""

    asset_id: str
    asset_role: Literal["uploaded_footage_source"]
    source_id: str
    source_uri: str
    file_name: str
    mime_type: str | None
    width: int
    height: int
    fps: float
    duration_seconds: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreviewPacket:
    """Narrow packet payload intended for future review/compiler handoff."""

    schema_version: str
    packet_kind: Literal["uploaded_footage_preview"]
    source_id: str
    transcript_ref: str
    cut_plan_ref: str
    caption_plan_ref: str
    scenes: tuple[PreviewPacketScene, ...]
    assets: tuple[PreviewPacketAsset, ...]


@dataclass(frozen=True)
class UploadedFootagePreviewBundle:
    """Top-level preview bundle artifact set."""

    schema_version: str
    source: UploadedFootageSource
    transcript: TranscriptArtifact
    cut_plan: CutPlanArtifact
    caption_plan: CaptionPlanArtifact
    preview_packet: PreviewPacket

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bundle into plain Python types."""
        return _json_ready(asdict(self))


def build_uploaded_footage_preview(
    source: UploadedFootageSource,
    *,
    language: str = "en",
    caption_style_preset: str = "logan_default",
) -> UploadedFootagePreviewBundle:
    """Build a preview bundle from uploaded footage metadata only."""
    _validate_source(source)

    transcript = _build_transcript_artifact(source, language=language)
    cut_plan = _build_cut_plan_artifact(source, transcript)
    caption_plan = _build_caption_plan_artifact(source, transcript, cut_plan, caption_style_preset)
    preview_packet = _build_preview_packet(source, transcript, cut_plan, caption_plan)

    return UploadedFootagePreviewBundle(
        schema_version=SCHEMA_VERSION,
        source=source,
        transcript=transcript,
        cut_plan=cut_plan,
        caption_plan=caption_plan,
        preview_packet=preview_packet,
    )


def _validate_source(source: UploadedFootageSource) -> None:
    if not source.source_id.strip():
        raise ValueError("source_id must be non-empty")
    if not source.source_uri.strip():
        raise ValueError("source_uri must be non-empty")
    if source.duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if source.width <= 0 or source.height <= 0:
        raise ValueError("width and height must be positive")
    if source.fps <= 0:
        raise ValueError("fps must be positive")

    previous_end = 0.0
    for segment in source.transcript_segments:
        if not segment.segment_id.strip():
            raise ValueError("transcript segment_id must be non-empty")
        if segment.start_seconds < 0 or segment.end_seconds <= segment.start_seconds:
            raise ValueError("transcript segment times must be increasing and positive")
        if segment.end_seconds > source.duration_seconds + 1e-6:
            raise ValueError("transcript segment exceeds source duration")
        if segment.start_seconds < previous_end - 1e-6:
            raise ValueError("transcript segments must be ordered by time")
        if not segment.text.strip():
            raise ValueError("transcript segment text must be non-empty")
        previous_end = segment.end_seconds


def _build_transcript_artifact(
    source: UploadedFootageSource,
    *,
    language: str,
) -> TranscriptArtifact:
    normalized_segments = tuple(
        UploadedFootageTranscriptSegment(
            segment_id=segment.segment_id,
            start_seconds=round(float(segment.start_seconds), 3),
            end_seconds=round(float(segment.end_seconds), 3),
            text=" ".join(segment.text.split()),
            speaker_label=segment.speaker_label,
            confidence=segment.confidence,
        )
        for segment in source.transcript_segments
    )
    full_text = " ".join(segment.text for segment in normalized_segments)
    return TranscriptArtifact(
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        source_id=source.source_id,
        full_text=full_text,
        language=language,
        segments=normalized_segments,
    )


def _build_cut_plan_artifact(
    source: UploadedFootageSource,
    transcript: TranscriptArtifact,
) -> CutPlanArtifact:
    if not transcript.segments:
        step = CutPlanStep(
            clip_id=f"{source.source_id}:clip:1",
            source_id=source.source_id,
            start_seconds=0.0,
            end_seconds=round(source.duration_seconds, 3),
            transcript_segment_ids=(),
            intent="body",
        )
        return CutPlanArtifact(
            schema_version=CUT_PLAN_SCHEMA_VERSION,
            source_id=source.source_id,
            total_duration_seconds=step.end_seconds - step.start_seconds,
            steps=(step,),
        )

    total_segments = len(transcript.segments)
    steps: list[CutPlanStep] = []
    for index, segment in enumerate(transcript.segments, start=1):
        intent: Literal["hook", "body", "cta", "bridge"] = "body"
        if index == 1:
            intent = "hook"
        elif index == total_segments:
            intent = "cta"
        elif total_segments > 2:
            intent = "bridge"

        clip_start = round(segment.start_seconds, 3)
        clip_end = round(max(segment.end_seconds, clip_start + MIN_CLIP_DURATION_SECONDS), 3)
        clip_end = min(clip_end, round(source.duration_seconds, 3))
        steps.append(
            CutPlanStep(
                clip_id=f"{source.source_id}:clip:{index}",
                source_id=source.source_id,
                start_seconds=clip_start,
                end_seconds=clip_end,
                transcript_segment_ids=(segment.segment_id,),
                intent=intent,
            )
        )

    total_duration = round(sum(step.end_seconds - step.start_seconds for step in steps), 3)
    return CutPlanArtifact(
        schema_version=CUT_PLAN_SCHEMA_VERSION,
        source_id=source.source_id,
        total_duration_seconds=total_duration,
        steps=tuple(steps),
    )


def _build_caption_plan_artifact(
    source: UploadedFootageSource,
    transcript: TranscriptArtifact,
    cut_plan: CutPlanArtifact,
    style_preset: str,
) -> CaptionPlanArtifact:
    transcript_by_id = {segment.segment_id: segment for segment in transcript.segments}
    cues: list[CaptionPlanCue] = []

    for index, step in enumerate(cut_plan.steps, start=1):
        step_segments = [
            transcript_by_id[segment_id]
            for segment_id in step.transcript_segment_ids
            if segment_id in transcript_by_id
        ]
        cue_text = " ".join(segment.text for segment in step_segments).strip()
        if not cue_text:
            cue_text = source.file_name

        cues.append(
            CaptionPlanCue(
                cue_id=f"{source.source_id}:caption:{index}",
                start_seconds=step.start_seconds,
                end_seconds=step.end_seconds,
                text=cue_text,
                transcript_segment_ids=step.transcript_segment_ids,
            )
        )

    return CaptionPlanArtifact(
        schema_version=CAPTION_PLAN_SCHEMA_VERSION,
        source_id=source.source_id,
        style_preset=style_preset,
        cues=tuple(cues),
    )


def _build_preview_packet(
    source: UploadedFootageSource,
    transcript: TranscriptArtifact,
    cut_plan: CutPlanArtifact,
    caption_plan: CaptionPlanArtifact,
) -> PreviewPacket:
    cue_ids_by_segment_id: dict[str, list[str]] = {}
    for cue in caption_plan.cues:
        for segment_id in cue.transcript_segment_ids:
            cue_ids_by_segment_id.setdefault(segment_id, []).append(cue.cue_id)

    transcript_by_id = {segment.segment_id: segment for segment in transcript.segments}
    scenes: list[PreviewPacketScene] = []
    for index, step in enumerate(cut_plan.steps, start=1):
        texts = [
            transcript_by_id[segment_id].text
            for segment_id in step.transcript_segment_ids
            if segment_id in transcript_by_id
        ]
        caption_cue_ids: list[str] = []
        for segment_id in step.transcript_segment_ids:
            caption_cue_ids.extend(cue_ids_by_segment_id.get(segment_id, []))

        scenes.append(
            PreviewPacketScene(
                scene_id=f"{source.source_id}:scene:{index}",
                scene_number=index,
                source_id=source.source_id,
                start_seconds=step.start_seconds,
                end_seconds=step.end_seconds,
                transcript_segment_ids=step.transcript_segment_ids,
                caption_cue_ids=tuple(dict.fromkeys(caption_cue_ids)),
                primary_text=" ".join(texts).strip() or source.file_name,
                asset_role="uploaded_footage",
            )
        )

    mime_type = source.metadata.get("mime_type")
    asset = PreviewPacketAsset(
        asset_id=f"{source.source_id}:asset:source",
        asset_role="uploaded_footage_source",
        source_id=source.source_id,
        source_uri=source.source_uri,
        file_name=source.file_name,
        mime_type=str(mime_type) if mime_type is not None else None,
        width=source.width,
        height=source.height,
        fps=source.fps,
        duration_seconds=source.duration_seconds,
        metadata=dict(source.metadata),
    )

    return PreviewPacket(
        schema_version=PACKET_SCHEMA_VERSION,
        packet_kind="uploaded_footage_preview",
        source_id=source.source_id,
        transcript_ref="transcript",
        cut_plan_ref="cut_plan",
        caption_plan_ref="caption_plan",
        scenes=tuple(scenes),
        assets=(asset,),
    )


def _json_ready(value: Any) -> Any:
    """Normalize dataclass output into JSON-friendly containers."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
