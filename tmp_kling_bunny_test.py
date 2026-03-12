import asyncio,sys,httpx
from pathlib import Path
sys.path.insert(0, r"C:\Users\Logan\Downloads\ai-shorts-engine\src")
from shorts_engine.adapters.video_gen.kling import KlingProvider
from shorts_engine.adapters.video_gen.base import VideoGenRequest

img = Path(r"C:\Users\Logan\.openclaw\media\inbound\file_147---51401cbf-1d66-45b4-80aa-d21b1082cb23.jpg")
out = Path(r"C:\Users\Logan\Downloads\ai-shorts-engine\output\samples\kling_bunny_test.mp4")

async def main():
    provider = KlingProvider()
    req = VideoGenRequest(
        prompt="pink bunny dancing energetically, meme style, full body visible, centered",
        duration_seconds=3,
        aspect_ratio='9:16',
        reference_images=[img.read_bytes()],
    )
    res = await provider.generate(req)
    print('success', res.success)
    print('error', res.error_message)
    print('meta', res.metadata)
    if res.success:
        url = (res.metadata or {}).get('video_url')
        if url:
            async with httpx.AsyncClient(timeout=240.0, follow_redirects=True) as c:
                r = await c.get(url)
                r.raise_for_status()
                out.write_bytes(r.content)
            print('saved', out, out.stat().st_size)

asyncio.run(main())
