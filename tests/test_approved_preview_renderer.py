"""Tests for the direct approved-preview renderer helpers."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from shorts_engine.cli import app
from shorts_engine.services.approved_preview_renderer import (
    _estimate_scene_durations,
    build_default_output_name,
    validate_approved_preview_inputs,
)


def test_validate_approved_preview_inputs_requires_two_scenes(tmp_path: Path):
    image_1 = tmp_path / "scene1.jpg"
    image_2 = tmp_path / "scene2.jpg"
    image_1.write_bytes(b"a")
    image_2.write_bytes(b"b")

    scenes = validate_approved_preview_inputs(
        "car",
        [str(image_1), str(image_2)],
        ["Good buy.", "Bad buy."],
    )

    assert len(scenes) == 2
    assert scenes[0].line == "Good buy."
    assert scenes[1].image_path == image_2


def test_validate_approved_preview_inputs_rejects_wrong_counts(tmp_path: Path):
    image_1 = tmp_path / "scene1.jpg"
    image_1.write_bytes(b"a")

    with pytest.raises(ValueError, match="exactly 2 --scene-image"):
        validate_approved_preview_inputs(
            "car",
            [str(image_1)],
            ["Only one.", "Still two lines."],
        )


def test_estimate_scene_durations_prefers_word_boundaries():
    boundaries = [
        {"text": "Good", "end_seconds": 0.4},
        {"text": "used", "end_seconds": 0.8},
        {"text": "SUVs", "end_seconds": 1.2},
        {"text": "first", "end_seconds": 1.7},
        {"text": "Avoid", "end_seconds": 2.2},
        {"text": "the", "end_seconds": 2.6},
        {"text": "money", "end_seconds": 3.1},
        {"text": "pit", "end_seconds": 3.6},
    ]

    durations = _estimate_scene_durations(
        ["Good used SUVs first.", "Avoid the money pit."],
        boundaries,
        3.6,
    )

    assert durations[0] == pytest.approx(1.7)
    assert durations[1] == pytest.approx(1.9)


def test_build_default_output_name_contains_brand():
    assert build_default_output_name("moatifi").startswith("moatifi_approved_preview_")


def test_preview_render_approved_validate_only(tmp_path: Path):
    image_1 = tmp_path / "scene1.jpg"
    image_2 = tmp_path / "scene2.jpg"
    image_1.write_bytes(b"a")
    image_2.write_bytes(b"b")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "preview",
            "render-approved",
            "--brand",
            "car",
            "--scene-image",
            str(image_1),
            "--scene-image",
            str(image_2),
            "--scene-line",
            "Buy this one.",
            "--scene-line",
            "Skip this one.",
            "--validate-only",
        ],
    )

    assert result.exit_code == 0
    assert "Approved Preview Render Plan" in result.stdout
    assert "Buy this one." in result.stdout
