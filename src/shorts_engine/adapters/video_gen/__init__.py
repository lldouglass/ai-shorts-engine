"""Video generation adapters."""

from shorts_engine.adapters.video_gen.base import (
    VideoGenProvider,
    VideoGenRequest,
    VideoGenResult,
)
from shorts_engine.adapters.video_gen.kling import KlingProvider
from shorts_engine.adapters.video_gen.luma import LumaProvider
from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider
from shorts_engine.adapters.video_gen.veo import VeoProvider

__all__ = [
    "VideoGenProvider",
    "VideoGenRequest",
    "VideoGenResult",
    "KlingProvider",
    "LumaProvider",
    "StubVideoGenProvider",
    "VeoProvider",
]
