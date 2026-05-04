"""Regression tests for standalone generation provider selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shorts_engine.adapters.video_gen.base import VideoGenProvider, VideoGenRequest, VideoGenResult


class FakeConfiguredVideoProvider(VideoGenProvider):
    def __init__(self) -> None:
        self.requests: list[VideoGenRequest] = []

    @property
    def name(self) -> str:
        return "fake-configured-provider"

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        self.requests.append(request)
        return VideoGenResult(success=True, metadata={})

    async def check_status(self, job_id: str) -> dict[str, Any]:
        return {"job_id": job_id, "status": "completed"}


class HardcodedVeoShouldNotBeUsed:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("generate_video.py must use the configured provider factory")


def _load_generate_video_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "generate_video.py"
    spec = importlib.util.spec_from_file_location("generate_video", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_generate_video_clips_uses_configured_provider_factory(monkeypatch, tmp_path) -> None:
    """Standalone clip generation must not bypass VIDEO_GEN_PROVIDER with hardcoded Veo."""
    generate_video = _load_generate_video_module()
    provider = FakeConfiguredVideoProvider()

    monkeypatch.setattr(generate_video, "get_video_gen_provider", lambda: provider)
    monkeypatch.setattr(generate_video, "VeoProvider", HardcodedVeoShouldNotBeUsed, raising=False)

    plan = SimpleNamespace(
        style_preset="DARK_DYSTOPIAN_ANIME",
        scenes=[
            SimpleNamespace(
                scene_number=1,
                caption_beat="Hero product reveal",
                visual_prompt="Crisp premium product macro shot",
                duration_seconds=6,
            )
        ],
    )

    await generate_video.generate_video_clips(
        plan,
        frame_chaining_enabled=False,
        output_dir=tmp_path,
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].prompt.startswith("Crisp premium product macro shot")
    assert provider.requests[0].aspect_ratio == "9:16"
