"""Video generation pipeline service."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from shorts_engine.adapters.analytics.base import AnalyticsAdapter
from shorts_engine.adapters.analytics.stub import StubAnalyticsAdapter
from shorts_engine.adapters.comments.base import CommentsAdapter
from shorts_engine.adapters.comments.stub import StubCommentsAdapter
from shorts_engine.adapters.publisher.base import PublisherAdapter
from shorts_engine.adapters.publisher.stub import StubPublisherAdapter
from shorts_engine.adapters.renderer.base import RendererProvider
from shorts_engine.adapters.renderer.stub import StubRendererProvider
from shorts_engine.adapters.video_gen.base import VideoGenProvider
from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider
from shorts_engine.config import settings
from shorts_engine.domain.enums import Platform
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result from running the video pipeline."""

    success: bool
    video_id: UUID | None = None
    publish_results: dict[Platform, dict[str, Any]] | None = None
    error_message: str | None = None


class PipelineService:
    """Orchestrates the end-to-end video generation pipeline.

    This service coordinates between different adapters to:
    1. Generate video from a prompt
    2. Render the video to final format
    3. Publish to configured platforms
    4. Track the video for analytics ingestion
    """

    def __init__(
        self,
        video_gen: VideoGenProvider | None = None,
        renderer: RendererProvider | None = None,
        publishers: dict[Platform, PublisherAdapter] | None = None,
        analytics: dict[Platform, AnalyticsAdapter] | None = None,
        comments: dict[Platform, CommentsAdapter] | None = None,
    ) -> None:
        """Initialize the pipeline with adapters.

        Args:
            video_gen: Video generation provider (defaults to stub)
            renderer: Video renderer provider (defaults to stub)
            publishers: Map of platforms to publisher adapters
            analytics: Map of platforms to analytics adapters
            comments: Map of platforms to comments adapters
        """
        self.video_gen = video_gen or self._get_video_gen_provider()
        self.renderer = renderer or self._get_renderer_provider()
        self.publishers = publishers or self._get_publishers()
        self.analytics = analytics or self._get_analytics_adapters()
        self.comments = comments or self._get_comments_adapters()

        logger.info(
            "pipeline_service_initialized",
            video_gen=self.video_gen.name,
            renderer=self.renderer.name,
            publishers=list(self.publishers.keys()),
        )

    def _get_video_gen_provider(self) -> VideoGenProvider:
        """Get the configured video generation provider."""
        provider_name = settings.video_gen_provider.lower()

        if provider_name == "stub":
            return StubVideoGenProvider()
        # Add other providers here as they're implemented
        # elif provider_name == "openai_sora":
        #     return OpenAISoraProvider()

        logger.warning(f"Unknown video_gen_provider '{provider_name}', using stub")
        return StubVideoGenProvider()

    def _get_renderer_provider(self) -> RendererProvider:
        """Get the configured renderer provider."""
        provider_name = settings.renderer_provider.lower()

        if provider_name == "stub":
            return StubRendererProvider()
        elif provider_name == "moviepy":
            from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer

            return MoviePyRenderer()
        elif provider_name == "creatomate":
            from shorts_engine.adapters.renderer.creatomate import CreatomateProvider

            return CreatomateProvider()

        logger.warning(f"Unknown renderer_provider '{provider_name}', using stub")
        return StubRendererProvider()

    def _get_publishers(self) -> dict[Platform, PublisherAdapter]:
        """Get configured publisher adapters."""
        publishers: dict[Platform, PublisherAdapter] = {}

        if settings.publisher_youtube_enabled:
            publishers[Platform.YOUTUBE] = StubPublisherAdapter(Platform.YOUTUBE)

        if settings.publisher_tiktok_enabled:
            publishers[Platform.TIKTOK] = StubPublisherAdapter(Platform.TIKTOK)

        if settings.publisher_instagram_enabled:
            publishers[Platform.INSTAGRAM] = StubPublisherAdapter(Platform.INSTAGRAM)

        # If no platforms enabled, add a default stub for testing
        if not publishers:
            publishers[Platform.YOUTUBE] = StubPublisherAdapter(Platform.YOUTUBE)

        return publishers

    def _get_analytics_adapters(self) -> dict[Platform, AnalyticsAdapter]:
        """Get analytics adapters for configured platforms."""
        return {platform: StubAnalyticsAdapter(platform) for platform in self.publishers}

    def _get_comments_adapters(self) -> dict[Platform, CommentsAdapter]:
        """Get comments adapters for configured platforms."""
        return {platform: StubCommentsAdapter(platform) for platform in self.publishers}

    async def health_check(self) -> dict[str, bool]:
        """Check health of all pipeline components."""
        results = {
            "video_gen": await self.video_gen.health_check(),
            "renderer": await self.renderer.health_check(),
        }

        for platform, publisher in self.publishers.items():
            results[f"publisher_{platform}"] = await publisher.health_check()

        for platform, analytics in self.analytics.items():
            results[f"analytics_{platform}"] = await analytics.health_check()

        return results
