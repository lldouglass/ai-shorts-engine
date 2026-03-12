"""Research pipeline tasks for content trend discovery.

These tasks run the content research engine to discover trending topics
and generate video ideas. Can be triggered manually or by the autonomous
nightly batch.
"""

import asyncio
import json
from typing import Any
from uuid import UUID

from shorts_engine.config import get_settings
from shorts_engine.logging import get_logger
from shorts_engine.worker import celery_app

logger = get_logger(__name__)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _get_llm_provider() -> Any:
    """Get the configured LLM provider."""
    settings = get_settings()

    if settings.llm_provider == "openai":
        from shorts_engine.adapters.llm import OpenAIProvider
        return OpenAIProvider(model=settings.openai_model)
    elif settings.llm_provider == "anthropic":
        from shorts_engine.adapters.llm import AnthropicProvider
        return AnthropicProvider()
    else:
        from shorts_engine.adapters.llm import StubLLMProvider
        return StubLLMProvider()


def _build_researcher() -> Any:
    """Build a ContentResearcher from config."""
    from shorts_engine.services.content_researcher import ContentResearcher

    settings = get_settings()
    llm = _get_llm_provider()

    competitor_channels = [
        ch.strip()
        for ch in settings.research_competitor_channels.split(",")
        if ch.strip()
    ]

    return ContentResearcher(
        llm_provider=llm,
        youtube_api_key=settings.google_api_key,
        competitor_channels=competitor_channels,
        tiktok_region=settings.research_tiktok_region,
    )


@celery_app.task(
    bind=True,
    name="research.run_trend_research",
    max_retries=2,
    default_retry_delay=120,
)
def run_trend_research_task(
    self: Any,
    n: int | None = None,
    categories: list[str] | None = None,
    recent_videos: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full content research pipeline.

    Fetches trends from TikTok + YouTube, synthesizes into video ideas,
    saves results to storage/research/.

    Args:
        n: Number of ideas to generate (default from config).
        categories: Optional category filters.
        recent_videos: Recent video titles to avoid duplicates.

    Returns:
        Dict with generated video ideas and metadata.
    """
    task_id = self.request.id
    settings = get_settings()

    if not settings.research_enabled:
        return {"success": False, "error": "Research is disabled. Set RESEARCH_ENABLED=true"}

    n = n or settings.research_ideas_per_batch

    # Parse categories from config if not provided
    if categories is None and settings.research_categories:
        categories = [c.strip() for c in settings.research_categories.split(",") if c.strip()]

    logger.info(
        "trend_research_started",
        task_id=task_id,
        n=n,
        categories=categories,
    )

    try:
        researcher = _build_researcher()
        ideas = _run_async(
            researcher.research_and_generate(
                n=n,
                categories=categories,
                recent_videos=recent_videos,
            )
        )

        # Convert to serializable format
        ideas_data = [idea.to_dict() for idea in ideas]

        logger.info(
            "trend_research_completed",
            task_id=task_id,
            ideas_generated=len(ideas_data),
            top_idea=ideas_data[0]["title"] if ideas_data else None,
            top_score=ideas_data[0]["overall_score"] if ideas_data else 0,
        )

        return {
            "success": True,
            "ideas": ideas_data,
            "count": len(ideas_data),
        }

    except Exception as e:
        logger.error("trend_research_failed", task_id=task_id, error=str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(
    bind=True,
    name="research.fetch_tiktok_trends",
    max_retries=3,
    default_retry_delay=60,
)
def fetch_tiktok_trends_task(
    self: Any,
    limit: int = 50,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch TikTok trends only (for debugging/monitoring).

    Args:
        limit: Max number of signals.
        categories: Optional category filters.

    Returns:
        Dict with raw trend signals.
    """
    from shorts_engine.adapters.research.tiktok import TikTokResearchProvider

    settings = get_settings()
    provider = TikTokResearchProvider(region=settings.research_tiktok_region)

    result = _run_async(provider.fetch_trends(categories=categories, limit=limit))

    signals_data = [
        {
            "title": s.title,
            "views": s.views,
            "virality_score": round(s.virality_score, 3),
            "category": s.category.value,
            "hashtags": s.hashtags,
            "url": s.url,
        }
        for s in result.signals
    ]

    return {
        "success": result.success,
        "source": "tiktok",
        "signals": signals_data,
        "count": len(signals_data),
        "error": result.error,
    }


@celery_app.task(
    bind=True,
    name="research.fetch_youtube_trends",
    max_retries=3,
    default_retry_delay=60,
)
def fetch_youtube_trends_task(
    self: Any,
    limit: int = 50,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch YouTube trends only (for debugging/monitoring).

    Args:
        limit: Max number of signals.
        categories: Optional category filters.

    Returns:
        Dict with raw trend signals.
    """
    settings = get_settings()

    from shorts_engine.adapters.research.youtube_trends import YouTubeResearchProvider

    competitor_channels = [
        ch.strip()
        for ch in settings.research_competitor_channels.split(",")
        if ch.strip()
    ]

    provider = YouTubeResearchProvider(
        api_key=settings.google_api_key,
        competitor_channels=competitor_channels,
    )

    result = _run_async(provider.fetch_trends(categories=categories, limit=limit))

    signals_data = [
        {
            "title": s.title,
            "views": s.views,
            "likes": s.likes,
            "virality_score": round(s.virality_score, 3),
            "category": s.category.value,
            "velocity": round(s.velocity, 1) if s.velocity else None,
            "url": s.url,
        }
        for s in result.signals
    ]

    return {
        "success": result.success,
        "source": "youtube",
        "signals": signals_data,
        "count": len(signals_data),
        "error": result.error,
    }
