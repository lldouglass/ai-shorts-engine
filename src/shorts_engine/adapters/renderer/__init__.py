"""Video rendering adapters."""

from shorts_engine.adapters.renderer.base import (
    RendererProvider,
    RenderRequest,
    RenderResult,
)
from shorts_engine.adapters.renderer.creatomate import (
    CreatomateProvider,
    CreatomateRenderRequest,
    ImageCompositionRequest,
    ImageSceneClip,
    MotionParams,
    SceneClip,
    build_creatomate_payload,
)
from shorts_engine.adapters.renderer.stub import StubRendererProvider

__all__ = [
    "RendererProvider",
    "RenderRequest",
    "RenderResult",
    "CreatomateProvider",
    "CreatomateRenderRequest",
    "ImageCompositionRequest",
    "ImageSceneClip",
    "MotionParams",
    "SceneClip",
    "StubRendererProvider",
    "build_creatomate_payload",
]
