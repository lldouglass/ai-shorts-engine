"""Content Research Engine - Scrapes trend data from TikTok and YouTube.

Architecture:
- SCRAPING is automated (TikTok Playwright + YouTube Data API)
- SYNTHESIS is done by Kade (Claude agent) who has context about the channel,
  AI video gen capabilities, and business goals. Not delegated to a generic LLM.

The scraper saves raw signals to storage/research/latest_signals.json.
Kade reads this file and generates video ideas with real strategic thinking.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shorts_engine.adapters.research.base import (
    ResearchResult,
    TrendSignal,
    TrendSource,
)
from shorts_engine.adapters.research.tiktok import TikTokResearchProvider
from shorts_engine.adapters.research.youtube_trends import YouTubeResearchProvider
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class ContentResearcher:
    """Scrapes trend signals from TikTok and YouTube.

    Handles DATA COLLECTION only. Synthesis (turning signals into
    video ideas) is done by Kade (Claude agent).

    Usage:
        researcher = ContentResearcher(...)
        signals = await researcher.scrape_all()
        # signals saved to storage/research/latest_signals.json
        # Kade reads this and generates ideas
    """

    def __init__(
        self,
        youtube_api_key: str | None = None,
        competitor_channels: list[str] | None = None,
        tiktok_region: str = "US",
        research_dir: str | Path | None = None,
    ):
        self._tiktok = TikTokResearchProvider(region=tiktok_region)
        self._youtube = YouTubeResearchProvider(
            api_key=youtube_api_key,
            competitor_channels=competitor_channels or [],
        )
        self._research_dir = Path(research_dir) if research_dir else Path("storage/research")
        self._research_dir.mkdir(parents=True, exist_ok=True)

    async def scrape_all(
        self,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scrape trend signals from all sources and save to disk.

        Returns raw signal data dict, also saved to latest_signals.json.
        """
        logger.info("content_research_scrape_started", categories=categories)

        results: dict[str, Any] = {
            "scraped_at": datetime.now(UTC).isoformat(),
        }

        # TikTok
        tiktok_result = await self._safe_fetch(
            self._tiktok.fetch_trends(categories=categories, limit=50),
            "tiktok",
        )
        if tiktok_result:
            results["tiktok"] = {
                "success": tiktok_result.success,
                "count": len(tiktok_result.signals),
                "signals": [self._signal_to_dict(s) for s in tiktok_result.signals],
            }

        # YouTube
        youtube_result = await self._safe_fetch(
            self._youtube.fetch_trends(categories=categories, limit=50),
            "youtube",
        )
        if youtube_result:
            results["youtube"] = {
                "success": youtube_result.success,
                "count": len(youtube_result.signals),
                "signals": [self._signal_to_dict(s) for s in youtube_result.signals],
            }

        # Competitor videos
        competitor_result = await self._safe_fetch(
            self._youtube.fetch_competitor_videos(limit=20),
            "youtube_competitors",
        )
        if competitor_result and competitor_result.signals:
            results["competitors"] = {
                "count": len(competitor_result.signals),
                "signals": [self._signal_to_dict(s) for s in competitor_result.signals],
            }

        total = sum(
            r.get("count", 0) for r in [
                results.get("tiktok", {}),
                results.get("youtube", {}),
                results.get("competitors", {}),
            ]
        )

        logger.info(
            "content_research_scrape_completed",
            tiktok=results.get("tiktok", {}).get("count", 0),
            youtube=results.get("youtube", {}).get("count", 0),
            competitors=results.get("competitors", {}).get("count", 0),
            total=total,
        )

        # Save to disk
        await self._save_signals(results)

        return results

    def _signal_to_dict(self, s: TrendSignal) -> dict[str, Any]:
        """Convert a TrendSignal to a serializable dict."""
        return {
            "title": s.title,
            "source": s.source.value,
            "views": s.views,
            "likes": s.likes,
            "comments": s.comments,
            "shares": s.shares,
            "engagement_rate": round(s.engagement_rate, 4),
            "hashtags": s.hashtags,
            "category": s.category.value,
            "creator": s.creator,
            "url": s.url,
            "velocity": round(s.velocity, 1) if s.velocity else None,
            "published_at": s.published_at.isoformat() if s.published_at else None,
            "description": s.description,
            "raw": {
                k: v for k, v in s.raw.items()
                if k in ("video_id", "is_short", "duration_seconds", "duration", "music", "competitor_channel_id")
            },
        }

    async def _safe_fetch(self, coro: Any, name: str) -> ResearchResult | None:
        """Safely fetch from a provider, catching all errors."""
        try:
            return await coro
        except Exception as e:
            logger.error("research_fetch_failed", provider=name, error=str(e))
            return ResearchResult(
                source=TrendSource.MANUAL,
                signals=[],
                error=str(e),
            )

    async def _save_signals(self, results: dict) -> None:
        """Save scraped signals to disk."""
        try:
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
            ts_file = self._research_dir / f"signals_{timestamp}.json"
            ts_file.write_text(
                json.dumps(results, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            latest_file = self._research_dir / "latest_signals.json"
            latest_file.write_text(
                json.dumps(results, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            logger.info("content_research_signals_saved", path=str(ts_file))

            # Cleanup old files (keep 30 days)
            cutoff = datetime.now(UTC).timestamp() - (30 * 86400)
            for f in self._research_dir.glob("signals_*.json"):
                if f.name != "latest_signals.json" and f.stat().st_mtime < cutoff:
                    f.unlink()

        except Exception as e:
            logger.warning("content_research_save_failed", error=str(e))

    async def close(self) -> None:
        """Clean up resources."""
        await self._tiktok.close()
        await self._youtube.close()
