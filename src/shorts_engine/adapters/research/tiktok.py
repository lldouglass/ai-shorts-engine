"""TikTok trend research provider.

Uses Playwright to load TikTok's explore page and intercept the internal
`/api/explore/item_list` API calls, which return full video metadata
including play counts, likes, comments, shares, and descriptions.

This is the most reliable approach since TikTok's public APIs are locked down
but their internal APIs fire normally when a real browser loads the page.
"""

import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from shorts_engine.adapters.research.base import (
    ContentCategory,
    ResearchProvider,
    ResearchResult,
    TrendSignal,
    TrendSource,
)
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}


def _classify_content(text: str, hashtags: list[str] | None = None) -> ContentCategory:
    """Classify content into a category based on text signals."""
    combined = f"{text} {' '.join(hashtags or [])}".lower()

    keywords = {
        ContentCategory.ANIME: ["anime", "manga", "otaku", "weeb", "naruto", "jujutsu", "demon slayer", "one piece", "dragonball"],
        ContentCategory.HORROR_DARK: ["horror", "scary", "creepy", "dark", "haunted", "paranormal", "ghost", "nightmare", "disturbing", "cursed"],
        ContentCategory.SCIFI_FANTASY: ["scifi", "sci-fi", "space", "alien", "future", "cyberpunk", "fantasy", "magic", "dragon"],
        ContentCategory.COMEDY: ["funny", "comedy", "humor", "joke", "lol", "meme", "hilarious", "prank", "funnyvideos"],
        ContentCategory.EDUCATION: ["learn", "howto", "tutorial", "explained", "facts", "science", "history", "didyouknow"],
        ContentCategory.GAMING: ["gaming", "gamer", "game", "esports", "fortnite", "minecraft", "roblox"],
        ContentCategory.TECH: ["tech", "ai", "coding", "programming", "gadget", "robot", "software"],
        ContentCategory.MOTIVATION: ["motivation", "grindset", "mindset", "success", "hustle", "inspire"],
        ContentCategory.STORYTELLING: ["storytime", "story", "pov", "narrative", "tale"],
        ContentCategory.NEWS_CURRENT: ["breaking", "news", "update"],
    }

    for category, kws in keywords.items():
        if any(kw in combined for kw in kws):
            return category

    return ContentCategory.ENTERTAINMENT


class TikTokResearchProvider(ResearchProvider):
    """TikTok trend research via Playwright network interception.

    Loads the TikTok explore page in a headless browser and intercepts
    the internal API calls that return full video metadata with engagement stats.
    """

    def __init__(self, region: str = "US", scroll_rounds: int = 3):
        """Initialize the provider.

        Args:
            region: Country code for regional trends.
            scroll_rounds: How many times to scroll the explore page
                          (each scroll loads ~8 more videos). More scrolls = more data
                          but slower. 3 rounds gets ~24 videos.
        """
        self._region = region
        self._scroll_rounds = scroll_rounds

    @property
    def name(self) -> str:
        return "tiktok"

    @property
    def source(self) -> TrendSource:
        return TrendSource.TIKTOK

    async def fetch_trends(
        self,
        categories: list[str] | None = None,
        limit: int = 50,
    ) -> ResearchResult:
        """Fetch trending TikTok content via Playwright interception."""
        logger.info("tiktok_research_started", region=self._region, limit=limit)

        signals: list[TrendSignal] = []

        # Primary: Playwright network interception
        try:
            playwright_signals = await self._fetch_via_playwright(limit)
            signals.extend(playwright_signals)
            logger.info("tiktok_playwright_signals", count=len(playwright_signals))
        except Exception as e:
            logger.warning("tiktok_playwright_failed", error=str(e))

        # Enrich with oEmbed data for any videos missing descriptions
        if signals:
            try:
                await self._enrich_with_oembed(signals)
            except Exception as e:
                logger.debug("tiktok_oembed_enrich_failed", error=str(e))

        # Filter by categories if specified
        if categories:
            cat_set = set(c.lower() for c in categories)
            signals = [s for s in signals if s.category.value in cat_set]

        # Deduplicate by video ID
        seen: set[str] = set()
        unique = []
        for s in signals:
            vid = s.raw.get("video_id", s.url or s.title)
            if vid not in seen:
                seen.add(vid)
                unique.append(s)
        signals = unique

        # Sort by views (most viral first)
        signals.sort(key=lambda s: s.views, reverse=True)
        signals = signals[:limit]

        result = ResearchResult(
            source=TrendSource.TIKTOK,
            signals=signals,
            error=None if signals else "No TikTok trends found",
        )

        logger.info(
            "tiktok_research_completed",
            total_signals=len(signals),
            success=result.success,
        )
        return result

    async def _fetch_via_playwright(self, limit: int) -> list[TrendSignal]:
        """Use Playwright to intercept TikTok's internal explore API."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("playwright_not_installed")
            return []

        captured_items: list[dict] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=HEADERS["User-Agent"],
            )
            page = await context.new_page()

            # Intercept API responses containing video data
            async def on_response(response):
                url = response.url
                if "api/explore/item_list" in url or "api/recommend/item_list" in url:
                    try:
                        ct = response.headers.get("content-type", "")
                        if "json" in ct:
                            body = await response.json()
                            items = body.get("itemList", [])
                            if items:
                                captured_items.extend(items)
                                logger.debug(
                                    "tiktok_api_intercepted",
                                    url=url[:100],
                                    items=len(items),
                                )
                    except Exception:
                        pass

            page.on("response", on_response)

            try:
                # Load explore page
                await page.goto(
                    "https://www.tiktok.com/explore",
                    wait_until="networkidle",
                    timeout=20000,
                )

                # Scroll to trigger more API loads
                for i in range(self._scroll_rounds):
                    await page.mouse.wheel(0, 2000)
                    await page.wait_for_timeout(2000)

                    # Stop if we have enough
                    if len(captured_items) >= limit:
                        break

            except Exception as e:
                logger.warning("tiktok_playwright_navigation", error=str(e))
            finally:
                await browser.close()

        logger.info("tiktok_items_captured", count=len(captured_items))

        # Convert raw items to TrendSignals
        signals = []
        for item in captured_items:
            signal = self._parse_item(item)
            if signal:
                signals.append(signal)

        return signals

    def _parse_item(self, item: dict) -> TrendSignal | None:
        """Parse a TikTok API item into a TrendSignal."""
        try:
            desc = item.get("desc", "")
            if not desc:
                return None

            stats = item.get("stats", {})
            author = item.get("author", {})
            video_id = item.get("id", "")
            author_id = author.get("uniqueId", "")

            # Extract hashtags from description
            hashtags = re.findall(r"#(\w+)", desc)

            # Calculate velocity (views per hour since publish)
            velocity = None
            create_time = item.get("createTime")
            play_count = stats.get("playCount", 0)
            if create_time and play_count:
                try:
                    pub_dt = datetime.fromtimestamp(int(create_time), tz=UTC)
                    hours_since = max((datetime.now(UTC) - pub_dt).total_seconds() / 3600, 1)
                    velocity = play_count / hours_since
                except (ValueError, TypeError, OSError):
                    pass

            published_at = None
            if create_time:
                try:
                    published_at = datetime.fromtimestamp(int(create_time), tz=UTC)
                except (ValueError, TypeError, OSError):
                    pass

            return TrendSignal(
                title=desc[:200],
                source=TrendSource.TIKTOK,
                url=f"https://www.tiktok.com/@{author_id}/video/{video_id}" if author_id and video_id else None,
                views=play_count,
                likes=stats.get("diggCount", 0),
                comments=stats.get("commentCount", 0),
                shares=stats.get("shareCount", 0),
                hashtags=hashtags[:10],
                category=_classify_content(desc, hashtags),
                description=desc[:500],
                creator=author_id,
                creator_followers=author.get("followerCount"),
                published_at=published_at,
                velocity=velocity,
                raw={
                    "video_id": video_id,
                    "author_id": author_id,
                    "duration": item.get("video", {}).get("duration", 0),
                    "music": item.get("music", {}).get("title", ""),
                },
            )
        except Exception as e:
            logger.debug("tiktok_parse_item_error", error=str(e))
            return None

    async def _enrich_with_oembed(self, signals: list[TrendSignal]) -> None:
        """Enrich signals with oEmbed data (author names, etc.)."""
        async with httpx.AsyncClient(timeout=10) as client:
            for signal in signals[:20]:  # Limit to avoid rate limiting
                if signal.url and not signal.creator:
                    try:
                        resp = await client.get(
                            "https://www.tiktok.com/oembed",
                            params={"url": signal.url},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if not signal.creator:
                                signal.creator = data.get("author_name", "")
                    except Exception:
                        pass

    async def close(self) -> None:
        """No persistent resources to close."""
        pass
