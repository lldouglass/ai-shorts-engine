"""Video rendering adapters."""

from shorts_engine.adapters.renderer.base import (
    RendererProvider,
    RenderRequest,
    RenderResult,
)
from shorts_engine.adapters.renderer.creatomate import (
    CreatomateProvider,
    CreatomateRenderRequest,
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
    "SceneClip",
    "StubRendererProvider",
    "build_creatomate_payload",
]
