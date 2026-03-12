"""YouTube trend research provider.

Uses the YouTube Data API v3 to:
1. Find trending Shorts in specific categories
2. Analyze competitor channels (top videos by views)
3. Search for rising content in target niches

Requires YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in config,
OR just an API key for public data access.
"""

from datetime import UTC, datetime, timedelta
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

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
YOUTUBE_TRENDING_URL = "https://www.googleapis.com/youtube/v3/videos"

# YouTube video category IDs for Shorts-friendly content
CATEGORY_IDS = {
    "entertainment": "24",
    "comedy": "23",
    "education": "27",
    "science_tech": "28",
    "gaming": "20",
    "film_animation": "1",
    "news": "25",
    "howto": "26",
    "music": "10",
}

# Map YouTube categories to our content categories
YT_CATEGORY_MAP = {
    "1": ContentCategory.SCIFI_FANTASY,  # Film & Animation
    "20": ContentCategory.GAMING,
    "23": ContentCategory.COMEDY,
    "24": ContentCategory.ENTERTAINMENT,
    "25": ContentCategory.NEWS_CURRENT,
    "26": ContentCategory.EDUCATION,
    "27": ContentCategory.EDUCATION,
    "28": ContentCategory.TECH,
}


def _classify_from_youtube(
    title: str,
    description: str,
    category_id: str | None,
    tags: list[str] | None = None,
) -> ContentCategory:
    """Classify YouTube content using category ID + text analysis."""
    # Try YouTube category first
    if category_id and category_id in YT_CATEGORY_MAP:
        return YT_CATEGORY_MAP[category_id]

    # Fall back to text classification
    from shorts_engine.adapters.research.base import ContentCategory
    text = f"{title} {description} {' '.join(tags or [])}".lower()

    keywords = {
        ContentCategory.ANIME: ["anime", "manga", "otaku"],
        ContentCategory.HORROR_DARK: ["horror", "scary", "creepy", "dark", "nightmare"],
        ContentCategory.SCIFI_FANTASY: ["scifi", "sci-fi", "space", "alien", "cyberpunk", "fantasy"],
        ContentCategory.COMEDY: ["funny", "comedy", "humor", "meme"],
        ContentCategory.EDUCATION: ["explained", "facts", "learn", "how to", "tutorial"],
        ContentCategory.GAMING: ["gaming", "game", "gameplay"],
        ContentCategory.TECH: ["tech", "ai", "coding", "robot"],
        ContentCategory.MOTIVATION: ["motivation", "grindset", "mindset", "success"],
        ContentCategory.STORYTELLING: ["storytime", "story", "pov"],
    }

    for category, kws in keywords.items():
        if any(kw in text for kw in kws):
            return category

    return ContentCategory.ENTERTAINMENT


class YouTubeResearchProvider(ResearchProvider):
    """YouTube trend research using Data API v3.

    Supports two auth modes (auto-detected):
    - API key mode: For public data (trending, search, channel videos)
    - OAuth mode: Uses refresh_token to get access_token (when API key isn't available
                  for the project with YouTube Data API enabled)
    """

    def __init__(
        self,
        api_key: str | None = None,
        region: str = "US",
        competitor_channels: list[str] | None = None,
    ):
        """Initialize YouTube research provider.

        Args:
            api_key: Google API key for YouTube Data API.
            region: Region code for trending content (US, GB, etc.)
            competitor_channels: List of channel IDs to analyze.
        """
        self._api_key = api_key
        self._region = region
        self._competitor_channels = competitor_channels or []
        self._client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None
        self._use_oauth: bool = False  # Set to True after first API key 403

    @property
    def name(self) -> str:
        return "youtube"

    @property
    def source(self) -> TrendSource:
        return TrendSource.YOUTUBE

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def _get_auth_params(self) -> dict[str, str]:
        """Get auth parameters for YouTube API requests.

        Returns either {"key": api_key} or {"access_token": token} dict
        to be merged into request params.
        """
        # Try API key first
        if self._api_key:
            return {"key": self._api_key}

        from shorts_engine.config import get_settings
        settings = get_settings()

        if settings.google_api_key:
            return {"key": settings.google_api_key}

        # Fall back to OAuth refresh token
        if settings.youtube_client_id and settings.youtube_client_secret:
            # Check if we have a refresh token in env
            import os
            refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
            if refresh_token:
                access_token = await self._get_access_token(
                    refresh_token,
                    settings.youtube_client_id,
                    settings.youtube_client_secret,
                )
                if access_token:
                    return {"access_token": access_token}

        raise ValueError(
            "No YouTube auth configured. Set GOOGLE_API_KEY or "
            "YOUTUBE_CLIENT_ID + YOUTUBE_CLIENT_SECRET + YOUTUBE_REFRESH_TOKEN in .env"
        )

    async def _get_access_token(
        self,
        refresh_token: str,
        client_id: str,
        client_secret: str,
    ) -> str | None:
        """Get a valid access token, refreshing if needed."""
        # Return cached token if still valid
        if self._access_token and self._token_expires_at:
            if datetime.now(UTC) < self._token_expires_at:
                return self._access_token

        client = await self._get_client()
        try:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )

            if response.status_code == 200:
                data = response.json()
                self._access_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                self._token_expires_at = datetime.now(UTC) + timedelta(seconds=expires_in - 60)
                logger.info("youtube_oauth_token_refreshed", expires_in=expires_in)
                return self._access_token
            else:
                logger.error(
                    "youtube_oauth_refresh_failed",
                    status=response.status_code,
                    body=response.text[:200],
                )
                return None
        except Exception as e:
            logger.error("youtube_oauth_refresh_error", error=str(e))
            return None

    async def fetch_trends(
        self,
        categories: list[str] | None = None,
        limit: int = 50,
    ) -> ResearchResult:
        """Fetch trending YouTube Shorts and related content.

        Combines:
        1. YouTube trending videos (filtered for Shorts-length)
        2. Search for trending Shorts in target categories
        """
        logger.info(
            "youtube_research_started",
            region=self._region,
            categories=categories,
            limit=limit,
        )

        signals: list[TrendSignal] = []

        # Strategy 1: Trending videos (Shorts-length)
        try:
            trending = await self._fetch_trending(limit=limit)
            signals.extend(trending)
            logger.info("youtube_trending_fetched", count=len(trending))
        except Exception as e:
            logger.warning("youtube_trending_failed", error=str(e))

        # Strategy 2: Search for popular Shorts in categories
        search_queries = self._build_search_queries(categories)
        for query in search_queries[:5]:  # Max 5 searches to conserve quota
            try:
                search_results = await self._search_shorts(query, limit=10)
                signals.extend(search_results)
            except Exception as e:
                logger.warning("youtube_search_failed", query=query, error=str(e))

        # Deduplicate by video ID
        seen_ids: set[str] = set()
        unique_signals = []
        for s in signals:
            vid_id = s.raw.get("video_id", s.url)
            if vid_id not in seen_ids:
                seen_ids.add(vid_id)
                unique_signals.append(s)

        # Sort by virality
        unique_signals.sort(key=lambda s: s.virality_score, reverse=True)
        unique_signals = unique_signals[:limit]

        result = ResearchResult(
            source=TrendSource.YOUTUBE,
            signals=unique_signals,
            error=None if unique_signals else "No YouTube trends found",
        )

        logger.info(
            "youtube_research_completed",
            total_signals=len(unique_signals),
            success=result.success,
        )

        return result

    async def fetch_competitor_videos(
        self,
        channel_ids: list[str] | None = None,
        limit: int = 50,
    ) -> ResearchResult:
        """Fetch top-performing videos from competitor channels.

        Args:
            channel_ids: Override competitor channel list. Uses configured list if None.
            limit: Max videos per channel.
        """
        channels = channel_ids or self._competitor_channels
        if not channels:
            return ResearchResult(
                source=TrendSource.YOUTUBE,
                signals=[],
                error="No competitor channels configured",
            )

        logger.info(
            "youtube_competitor_analysis_started",
            channel_count=len(channels),
        )

        signals: list[TrendSignal] = []

        for channel_id in channels:
            try:
                channel_signals = await self._fetch_channel_top_videos(
                    channel_id, limit=limit
                )
                signals.extend(channel_signals)
            except Exception as e:
                logger.warning(
                    "youtube_competitor_fetch_failed",
                    channel_id=channel_id,
                    error=str(e),
                )

        signals.sort(key=lambda s: s.views, reverse=True)

        return ResearchResult(
            source=TrendSource.YOUTUBE,
            signals=signals[:limit * len(channels)],
        )

    async def _fetch_trending(self, limit: int = 50) -> list[TrendSignal]:
        """Fetch YouTube trending videos, filtered for Shorts-length content."""
        response = await self._make_yt_request(
            YOUTUBE_TRENDING_URL,
            params={
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": self._region,
                "maxResults": min(limit, 50),
            },
        )

        if response.status_code != 200:
            logger.error("youtube_trending_api_error", status=response.status_code, body=response.text[:300])
            return []

        data = response.json()
        signals = []

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            # Parse duration to check if it's Shorts-length (< 180 seconds)
            duration = content.get("duration", "")
            duration_secs = self._parse_duration(duration)

            # We want Shorts (< 3 min) but also record longer trending content
            # as topic signals
            is_short = duration_secs > 0 and duration_secs <= 180

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            # Calculate velocity if we have publish time
            published = snippet.get("publishedAt")
            velocity = None
            if published:
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    hours_since = max((datetime.now(UTC) - pub_dt).total_seconds() / 3600, 1)
                    velocity = views / hours_since
                except (ValueError, TypeError):
                    pass

            signal = TrendSignal(
                title=snippet.get("title", ""),
                source=TrendSource.YOUTUBE,
                url=f"https://youtube.com/shorts/{item['id']}" if is_short else f"https://youtube.com/watch?v={item['id']}",
                views=views,
                likes=likes,
                comments=comments,
                hashtags=snippet.get("tags", [])[:10],
                category=_classify_from_youtube(
                    snippet.get("title", ""),
                    snippet.get("description", ""),
                    snippet.get("categoryId"),
                    snippet.get("tags"),
                ),
                description=snippet.get("description", "")[:500],
                creator=snippet.get("channelTitle", ""),
                published_at=datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None,
                velocity=velocity,
                raw={
                    "video_id": item["id"],
                    "is_short": is_short,
                    "duration_seconds": duration_secs,
                    "category_id": snippet.get("categoryId"),
                },
            )
            signals.append(signal)

        return signals

    async def _make_yt_request(self, url: str, params: dict) -> httpx.Response:
        """Make an authenticated YouTube API request.

        Tries API key first, falls back to OAuth on 403 (wrong project).
        Caches the decision so subsequent requests skip the failed method.
        """
        client = await self._get_client()

        # If we already know API key doesn't work, go straight to OAuth
        if self._use_oauth:
            oauth_token = await self._try_oauth_fallback()
            if oauth_token:
                return await client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {oauth_token}"},
                )

        auth = await self._get_auth_params()

        headers = {}
        if "access_token" in auth:
            headers["Authorization"] = f"Bearer {auth['access_token']}"
        else:
            params = {**params, **auth}

        response = await client.get(url, params=params, headers=headers)

        # If API key failed with 403, switch to OAuth permanently for this session
        if response.status_code == 403 and "key" in auth and not headers:
            logger.info("youtube_api_key_403_switching_to_oauth")
            self._use_oauth = True
            oauth_token = await self._try_oauth_fallback()
            if oauth_token:
                clean_params = {k: v for k, v in params.items() if k != "key"}
                response = await client.get(
                    url,
                    params=clean_params,
                    headers={"Authorization": f"Bearer {oauth_token}"},
                )

        return response

    async def _try_oauth_fallback(self) -> str | None:
        """Try to get an OAuth access token as fallback."""
        import os
        from shorts_engine.config import get_settings
        settings = get_settings()

        refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
        if refresh_token and settings.youtube_client_id and settings.youtube_client_secret:
            return await self._get_access_token(
                refresh_token,
                settings.youtube_client_id,
                settings.youtube_client_secret,
            )
        return None

    async def _search_shorts(self, query: str, limit: int = 10) -> list[TrendSignal]:
        """Search YouTube for Shorts matching a query."""
        # Search for recent, popular short videos
        published_after = (datetime.now(UTC) - timedelta(days=7)).isoformat()

        response = await self._make_yt_request(
            YOUTUBE_SEARCH_URL,
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "videoDuration": "short",  # Under 4 minutes
                "order": "viewCount",
                "publishedAfter": published_after,
                "regionCode": self._region,
                "maxResults": min(limit, 50),
            },
        )

        if response.status_code != 200:
            logger.warning("youtube_search_api_error", status=response.status_code, query=query)
            return []

        data = response.json()
        video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item.get("id", {})]

        if not video_ids:
            return []

        # Get full statistics for these videos
        return await self._get_video_details(video_ids)

    async def _fetch_channel_top_videos(
        self, channel_id: str, limit: int = 20
    ) -> list[TrendSignal]:
        """Fetch top-performing recent videos from a specific channel."""
        # Search within channel, sorted by view count
        published_after = (datetime.now(UTC) - timedelta(days=30)).isoformat()

        response = await self._make_yt_request(
            YOUTUBE_SEARCH_URL,
            params={
                "part": "snippet",
                "channelId": channel_id,
                "type": "video",
                "videoDuration": "short",
                "order": "viewCount",
                "publishedAfter": published_after,
                "maxResults": min(limit, 50),
            },
        )

        if response.status_code != 200:
            logger.warning(
                "youtube_channel_search_error",
                channel_id=channel_id,
                status=response.status_code,
            )
            return []

        data = response.json()
        video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item.get("id", {})]

        if not video_ids:
            return []

        signals = await self._get_video_details(video_ids)

        # Tag with competitor source
        for s in signals:
            s.raw["competitor_channel_id"] = channel_id

        return signals

    async def _get_video_details(self, video_ids: list[str]) -> list[TrendSignal]:
        """Get detailed statistics for a list of video IDs."""
        response = await self._make_yt_request(
            YOUTUBE_VIDEOS_URL,
            params={
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(video_ids),
            },
        )

        if response.status_code != 200:
            return []

        data = response.json()
        signals = []

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            duration_secs = self._parse_duration(content.get("duration", ""))
            is_short = 0 < duration_secs <= 180

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            published = snippet.get("publishedAt")
            velocity = None
            if published:
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    hours_since = max((datetime.now(UTC) - pub_dt).total_seconds() / 3600, 1)
                    velocity = views / hours_since
                except (ValueError, TypeError):
                    pass

            signal = TrendSignal(
                title=snippet.get("title", ""),
                source=TrendSource.YOUTUBE,
                url=f"https://youtube.com/shorts/{item['id']}" if is_short else f"https://youtube.com/watch?v={item['id']}",
                views=views,
                likes=likes,
                comments=comments,
                hashtags=snippet.get("tags", [])[:10],
                category=_classify_from_youtube(
                    snippet.get("title", ""),
                    snippet.get("description", ""),
                    snippet.get("categoryId"),
                    snippet.get("tags"),
                ),
                description=snippet.get("description", "")[:500],
                creator=snippet.get("channelTitle", ""),
                published_at=datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None,
                velocity=velocity,
                raw={
                    "video_id": item["id"],
                    "is_short": is_short,
                    "duration_seconds": duration_secs,
                    "category_id": snippet.get("categoryId"),
                },
            )
            signals.append(signal)

        return signals

    def _build_search_queries(self, categories: list[str] | None) -> list[str]:
        """Build YouTube search queries for Shorts discovery."""
        base_queries = [
            "#shorts viral",
            "AI animation shorts",
            "dark storytelling shorts",
            "scary AI shorts",
            "anime shorts viral",
            "motivational shorts",
            "mind blowing facts shorts",
            "sci-fi shorts",
        ]

        if categories:
            category_queries = [f"{cat} shorts viral" for cat in categories]
            return category_queries + base_queries
        return base_queries

    @staticmethod
    def _parse_duration(duration: str) -> int:
        """Parse ISO 8601 duration (PT1M30S) to seconds."""
        import re
        if not duration:
            return 0

        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
