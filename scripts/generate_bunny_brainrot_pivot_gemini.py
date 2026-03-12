"""Generate bunny brainrot intro + stock education pivot video (Gemini voice).

Flow:
1) Bunny dance brainrot hook (reference image guided)
2) Hard pivot text
3) Stock education visual
4) Comment CTA

Note: Do not bake copyrighted song into export. Add official track in-app.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
import wave
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shorts_engine.adapters.renderer.creatomate import CreatomateRenderRequest, SceneClip
from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer
from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.veo import VeoProvider
from shorts_engine.jobs.render_pipeline import _resolve_background_music_defaults

# User-provided bunny image from Telegram inbound media.
BUNNY_IMAGE = Path(
    r"C:\Users\Logan\.openclaw\media\inbound\file_147---51401cbf-1d66-45b4-80aa-d21b1082cb23.jpg"
)

HOOK_PROMPT = (
    "Pink bunny character dancing in a chaotic viral brainrot internet style, "
    "high-energy meme movement, exaggerated dance poses, colorful flashing lights, "
    "funny and absurd but polished cinematic quality, vertical 9:16"
)

EDU_PROMPT = (
    "Talking soda can character explaining stock investing in a simple way, "
    "clean modern desk setup with stock chart overlays, trustworthy educational style, "
    "clear visual storytelling, cinematic, vertical 9:16"
)

VOICE_TEXT = (
    "You just got caught brain rotting. Good. "
    "Now use that attention to learn one stock rule. "
    "Do not buy hype, buy pricing power. "
    "If a company can raise prices and customers still buy, that is a moat. "
    "Comment a ticker and I will rate the moat for you."
)


def _load_reference_image_bytes() -> list[bytes] | None:
    if not BUNNY_IMAGE.exists():
        print(f"Reference image missing: {BUNNY_IMAGE}")
        return None
    return [BUNNY_IMAGE.read_bytes()]


async def _generate_veo_clip(
    provider: VeoProvider,
    prompt: str,
    out_path: Path,
    reference_images: list[bytes] | None = None,
) -> Path:
    """Generate one Veo clip with retry."""
    last_error: str | None = None

    for attempt in range(1, 4):
        req = VideoGenRequest(
            prompt=prompt,
            duration_seconds=6,
            aspect_ratio="9:16",
            reference_images=reference_images,
        )
        result = await provider.generate(req)
        if not result.success:
            last_error = result.error_message or "unknown error"
            print(f"Veo attempt {attempt} failed: {last_error}")
            continue

        video_url = (result.metadata or {}).get("video_url")
        headers = (result.metadata or {}).get("download_headers", {})
        if not video_url:
            last_error = "no video URL returned"
            print(f"Veo attempt {attempt} failed: {last_error}")
            continue

        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            resp = await client.get(video_url, headers=headers)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)

        return out_path

    raise RuntimeError(f"Veo generation failed after retries: {last_error}")


async def _generate_gemini_tts_wav(text: str, out_wav: Path) -> Path:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY missing")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-tts:generateContent?key={key}"
    )

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Kore"}
                }
            },
        },
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        part = resp.json()["candidates"][0]["content"]["parts"][0]["inlineData"]

    raw = base64.b64decode(part["data"])
    mime = part.get("mimeType", "")
    rate = 24000
    m = re.search(r"rate=(\d+)", mime)
    if m:
        rate = int(m.group(1))

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw)

    return out_wav


async def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    hook_clip = out_dir / f"bunny_hook_raw_{ts}.mp4"
    edu_clip = out_dir / f"bunny_edu_raw_{ts}.mp4"
    wav_path = out_dir / f"bunny_pivot_voice_{ts}.wav"

    print("Generating hook clip (bunny brainrot)...")
    veo = VeoProvider()
    if not await veo.health_check():
        raise RuntimeError("Veo health check failed")

    ref_images = _load_reference_image_bytes()
    await _generate_veo_clip(veo, HOOK_PROMPT, hook_clip, reference_images=ref_images)

    print("Generating education clip...")
    await _generate_veo_clip(veo, EDU_PROMPT, edu_clip, reference_images=None)

    print("Generating Gemini voice...")
    await _generate_gemini_tts_wav(VOICE_TEXT, wav_path)

    bgm_url, bgm_volume = _resolve_background_music_defaults(None, None)

    scenes = [
        SceneClip(
            video_url=str(hook_clip),
            duration_seconds=2.4,
            caption_text="YOU JUST GOT CAUGHT",
            scene_number=1,
        ),
        SceneClip(
            video_url=str(hook_clip),
            duration_seconds=0.9,
            caption_text="BRAIN ROTTING",
            scene_number=2,
        ),
        SceneClip(
            video_url=str(edu_clip),
            duration_seconds=8.4,
            caption_text="BUY PRICING POWER, NOT HYPE",
            scene_number=3,
        ),
        SceneClip(
            video_url=str(edu_clip),
            duration_seconds=3.7,
            caption_text="COMMENT A TICKER FOR A MOAT SCORE",
            scene_number=4,
        ),
    ]

    print("Rendering final video...")
    renderer = MoviePyRenderer(output_dir=out_dir)
    result = await renderer.render_composition(
        CreatomateRenderRequest(
            scenes=scenes,
            voiceover_url=str(wav_path),
            background_music_url=bgm_url,
            background_music_volume=bgm_volume,
            width=1080,
            height=1920,
            fps=30,
        )
    )

    if not result.success or not result.output_path:
        raise RuntimeError(f"Render failed: {result.error_message}")

    final_path = out_dir / f"bunny_brainrot_pivot_{ts}.mp4"
    final_path.write_bytes(Path(result.output_path).read_bytes())
    print(f"Final video: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
