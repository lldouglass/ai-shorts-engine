"""Tests for uploaded footage preview bundle building."""

from shorts_engine.services.footage_preview_builder import (
    CAPTION_PLAN_SCHEMA_VERSION,
    CUT_PLAN_SCHEMA_VERSION,
    PACKET_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TRANSCRIPT_SCHEMA_VERSION,
    UploadedFootageSource,
    UploadedFootageTranscriptSegment,
    build_uploaded_footage_preview,
)


def _build_fixture_source() -> UploadedFootageSource:
    return UploadedFootageSource(
        source_id="logan-footage-001",
        source_uri="uploads/logan/raw/logan-footage-001.mp4",
        file_name="logan-footage-001.mp4",
        duration_seconds=12.0,
        width=1080,
        height=1920,
        fps=29.97,
        metadata={
            "mime_type": "video/mp4",
            "upload_session_id": "upload-123",
            "review_title": "Logan used-car rant",
        },
        transcript_segments=(
            UploadedFootageTranscriptSegment(
                segment_id="seg-1",
                start_seconds=0.0,
                end_seconds=2.5,
                text="This SUV looks cheap up front.",
                speaker_label="Logan",
                confidence=0.98,
            ),
            UploadedFootageTranscriptSegment(
                segment_id="seg-2",
                start_seconds=2.5,
                end_seconds=6.4,
                text="But one transmission repair can kill the deal.",
                speaker_label="Logan",
                confidence=0.97,
            ),
            UploadedFootageTranscriptSegment(
                segment_id="seg-3",
                start_seconds=6.4,
                end_seconds=10.8,
                text="Check the service history before you buy.",
                speaker_label="Logan",
                confidence=0.99,
            ),
        ),
    )


def test_build_uploaded_footage_preview_bundle_has_stable_artifacts():
    source = _build_fixture_source()

    bundle = build_uploaded_footage_preview(source)

    assert bundle.schema_version == SCHEMA_VERSION
    assert bundle.transcript.schema_version == TRANSCRIPT_SCHEMA_VERSION
    assert bundle.cut_plan.schema_version == CUT_PLAN_SCHEMA_VERSION
    assert bundle.caption_plan.schema_version == CAPTION_PLAN_SCHEMA_VERSION
    assert bundle.preview_packet.schema_version == PACKET_SCHEMA_VERSION
    assert bundle.transcript.full_text == (
        "This SUV looks cheap up front. "
        "But one transmission repair can kill the deal. "
        "Check the service history before you buy."
    )
    assert [step.intent for step in bundle.cut_plan.steps] == ["hook", "bridge", "cta"]
    assert bundle.caption_plan.cues[0].text == "This SUV looks cheap up front."
    assert bundle.preview_packet.scenes[0].asset_role == "uploaded_footage"
    assert bundle.preview_packet.assets[0].asset_role == "uploaded_footage_source"


def test_build_uploaded_footage_preview_bundle_serializes_packet_friendly_shape():
    source = _build_fixture_source()

    payload = build_uploaded_footage_preview(source).to_dict()

    assert payload["source"]["source_uri"] == "uploads/logan/raw/logan-footage-001.mp4"
    assert payload["transcript"]["segments"][1]["segment_id"] == "seg-2"
    assert payload["cut_plan"]["steps"][2]["transcript_segment_ids"] == ["seg-3"]
    assert payload["caption_plan"]["cues"][1]["cue_id"] == "logan-footage-001:caption:2"
    assert payload["preview_packet"]["packet_kind"] == "uploaded_footage_preview"
    assert payload["preview_packet"]["transcript_ref"] == "transcript"
    assert payload["preview_packet"]["scenes"][2]["caption_cue_ids"] == [
        "logan-footage-001:caption:3"
    ]


def test_build_uploaded_footage_preview_bundle_without_transcript_uses_single_clip():
    source = UploadedFootageSource(
        source_id="logan-broll-001",
        source_uri="uploads/logan/raw/logan-broll-001.mov",
        file_name="logan-broll-001.mov",
        duration_seconds=4.2,
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=False,
        metadata={"mime_type": "video/quicktime"},
    )

    bundle = build_uploaded_footage_preview(source)

    assert bundle.transcript.full_text == ""
    assert len(bundle.cut_plan.steps) == 1
    assert bundle.cut_plan.steps[0].start_seconds == 0.0
    assert bundle.cut_plan.steps[0].end_seconds == 4.2
    assert bundle.caption_plan.cues[0].text == "logan-broll-001.mov"
    assert bundle.preview_packet.scenes[0].primary_text == "logan-broll-001.mov"
