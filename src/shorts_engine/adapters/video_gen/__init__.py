"""Video generation adapters."""

from shorts_engine.adapters.video_gen.base import (
    VideoGenProvider,
    VideoGenRequest,
    VideoGenResult,
)
from shorts_engine.adapters.video_gen.luma import LumaProvider
from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider

__all__ = [
    "VideoGenProvider",
    "VideoGenRequest",
    "VideoGenResult",
    "LumaProvider",
    "StubVideoGenProvider",
]
