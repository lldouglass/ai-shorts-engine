"""Generate a single Seedance sample hook clip and save locally.

Usage:
  python scripts/generate_seedance_sample.py
  python scripts/generate_seedance_sample.py "your prompt"
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import httpx

# Ensure local src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.seedance import SeedanceProvider


DEFAULT_PROMPT = (
    "Hook shot for finance short video: dramatic push-in on glowing stock screens, "
    "neon data streams and candlestick charts, cinematic contrast lighting, "
    "high energy, premium commercial look, hyper-detailed, vertical composition"
)


async def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT

    provider = SeedanceProvider()
    if not await provider.health_check():
        print("SEEDANCE health check failed or key missing")
        return 1

    req = VideoGenRequest(
        prompt=prompt,
        duration_seconds=5,
        aspect_ratio="9:16",
    )

    print("Generating Seedance sample clip...")
    print(f"Prompt: {prompt}")

    result = await provider.generate(req)
    if not result.success:
        print(f"Generation failed: {result.error_message}")
        return 1

    video_url = (result.metadata or {}).get("video_url")
    task_id = (result.metadata or {}).get("task_id")
    if not video_url:
        print("No video URL returned")
        return 1

    out_dir = ROOT / "output" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"seedance_hook_{ts}.mp4"

    print(f"Task ID: {task_id}")
    print("Downloading clip...")
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        r = await client.get(video_url)
        r.raise_for_status()
        out_path.write_bytes(r.content)

    print(f"Saved: {out_path}")
    print(f"Bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
