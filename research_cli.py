"""Content Research CLI - Scrapes trends, saves raw data.

Synthesis is done by Kade (Claude agent), not by a generic LLM.

Usage:
    python research_cli.py                  # Scrape TikTok + YouTube, save signals
    python research_cli.py --tiktok-only    # Just TikTok
    python research_cli.py --youtube-only   # Just YouTube
    python research_cli.py --competitors UCxxx,UCyyy  # Include competitor analysis
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()


async def scrape_tiktok(region: str = "US", limit: int = 50) -> dict:
    """Scrape TikTok trends via Playwright."""
    from shorts_engine.adapters.research.tiktok import TikTokResearchProvider

    provider = TikTokResearchProvider(region=region)
    result = await provider.fetch_trends(limit=limit)
    await provider.close()

    signals = []
    for s in result.signals:
        signals.append({
            "title": s.title,
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
            "duration": s.raw.get("duration", 0),
            "music": s.raw.get("music", ""),
        })

    return {
        "source": "tiktok",
        "success": result.success,
        "count": len(signals),
        "signals": sorted(signals, key=lambda x: -x["views"]),
    }


async def scrape_youtube(
    competitors: list[str] | None = None,
    limit: int = 50,
) -> dict:
    """Scrape YouTube trends via Data API."""
    from shorts_engine.adapters.research.youtube_trends import YouTubeResearchProvider

    provider = YouTubeResearchProvider(competitor_channels=competitors or [])
    
    # Trending videos
    result = await provider.fetch_trends(limit=limit)
    
    signals = []
    for s in result.signals:
        signals.append({
            "title": s.title,
            "views": s.views,
            "likes": s.likes,
            "comments": s.comments,
            "engagement_rate": round(s.engagement_rate, 4),
            "hashtags": s.hashtags[:5],
            "category": s.category.value,
            "creator": s.creator,
            "url": s.url,
            "is_short": s.raw.get("is_short", False),
            "duration_seconds": s.raw.get("duration_seconds", 0),
            "velocity": round(s.velocity, 1) if s.velocity else None,
            "published_at": s.published_at.isoformat() if s.published_at else None,
        })

    # Competitor analysis
    comp_signals = []
    if competitors:
        comp_result = await provider.fetch_competitor_videos(limit=20)
        for s in comp_result.signals:
            comp_signals.append({
                "title": s.title,
                "views": s.views,
                "likes": s.likes,
                "creator": s.creator,
                "url": s.url,
                "is_short": s.raw.get("is_short", False),
                "category": s.category.value,
                "competitor_channel": s.raw.get("competitor_channel_id", ""),
            })

    await provider.close()

    return {
        "source": "youtube",
        "success": result.success,
        "count": len(signals),
        "signals": sorted(signals, key=lambda x: -x["views"]),
        "competitor_videos": sorted(comp_signals, key=lambda x: -x["views"]),
    }


async def run_research(
    tiktok: bool = True,
    youtube: bool = True,
    competitors: list[str] | None = None,
    region: str = "US",
) -> dict:
    """Run full research scrape and save results."""
    results = {
        "scraped_at": datetime.now(UTC).isoformat(),
        "region": region,
    }

    if tiktok:
        print("  Scraping TikTok trends (Playwright)...")
        results["tiktok"] = await scrape_tiktok(region=region)
        print(f"  Found {results['tiktok']['count']} TikTok signals")

    if youtube:
        print("  Scraping YouTube trends (OAuth)...")
        results["youtube"] = await scrape_youtube(competitors=competitors)
        print(f"  Found {results['youtube']['count']} YouTube signals")
        if competitors:
            comp_count = len(results["youtube"].get("competitor_videos", []))
            print(f"  Found {comp_count} competitor videos")

    # Save to file
    output_dir = Path("storage/research")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save timestamped version
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    output_file = output_dir / f"signals_{timestamp}.json"
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Also save as "latest" for easy access
    latest_file = output_dir / "latest_signals.json"
    latest_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n  Saved to: {output_file}")
    print(f"  Latest:   {latest_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Content Research Scraper")
    parser.add_argument("--tiktok-only", action="store_true")
    parser.add_argument("--youtube-only", action="store_true")
    parser.add_argument("--competitors", type=str, default="")
    parser.add_argument("--region", type=str, default="US")

    args = parser.parse_args()
    competitors = [c.strip() for c in args.competitors.split(",") if c.strip()]

    tiktok = not args.youtube_only
    youtube = not args.tiktok_only

    print(f"\n{'='*50}")
    print(f"  Content Research Scraper")
    print(f"{'='*50}\n")

    results = asyncio.run(run_research(
        tiktok=tiktok,
        youtube=youtube,
        competitors=competitors,
        region=args.region,
    ))

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}\n")

    if "tiktok" in results:
        tt = results["tiktok"]
        print(f"  TikTok: {tt['count']} signals")
        for s in tt["signals"][:5]:
            print(f"    {s['views']:>15,} views | {s['title'][:60]}")

    if "youtube" in results:
        yt = results["youtube"]
        print(f"\n  YouTube: {yt['count']} signals")
        for s in yt["signals"][:5]:
            short = "[SHORT]" if s.get("is_short") else "[VIDEO]"
            print(f"    {s['views']:>15,} views | {short} {s['title'][:55]}")

    print(f"\n  Raw data saved. Feed to Kade for synthesis.\n")


if __name__ == "__main__":
    main()
