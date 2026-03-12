"""Generate a Munger-style educational short with a talking candle (no bunny hook)."""

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

PROMPT_1 = (
    "Anthropomorphic talking candle, expressive face and subtle mouth movement, "
    "speaking directly to camera in a dark wood study, warm cinematic lighting, "
    "educational tone, high detail, vertical 9:16"
)

PROMPT_2 = (
    "Same talking candle character on a desk with simple stock chart overlays in the background, "
    "calm confident explanation vibe, cinematic close-up, warm practical atmosphere, vertical 9:16"
)

VOICE_TEXT = (
    "Most investors lose money the same way. "
    "They chase stories and ignore business quality. "
    "Charlie Munger's rule is simple, avoid stupid. "
    "If a company cannot raise prices without losing customers, the moat is weak. "
    "If profits depend on perfect conditions, skip it. "
    "Buy durable businesses and let compounding work. "
    "Drop a ticker and I will score the moat."
)


def _get_background_track() -> str | None:
    candidates = [
        ROOT / "ambient_pad.wav",
        ROOT / "storage" / "FEEL_IT_clip.mp3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


async def _generate_veo_clip(provider: VeoProvider, prompt: str, out_path: Path) -> Path:
    last_error: str | None = None

    for attempt in range(1, 4):
        req = VideoGenRequest(
            prompt=prompt,
            duration_seconds=6,
            aspect_ratio="9:16",
        )
        result = await provider.generate(req)
        if not result.success:
            last_error = result.error_message or "unknown error"
            print(f"Veo attempt {attempt} failed: {last_error}")
            continue

        video_url = (result.metadata or {}).get("video_url")
        headers = (result.metadata or {}).get("download_headers", {})
        if not video_url:
            last_error = "Veo returned no video URL"
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

    clip1 = out_dir / f"candle_clip1_{ts}.mp4"
    clip2 = out_dir / f"candle_clip2_{ts}.mp4"
    voice = out_dir / f"candle_voice_{ts}.wav"

    veo = VeoProvider()
    if not await veo.health_check():
        raise RuntimeError("Veo health check failed")

    print("Generating candle scene 1...")
    await _generate_veo_clip(veo, PROMPT_1, clip1)

    print("Generating candle scene 2...")
    await _generate_veo_clip(veo, PROMPT_2, clip2)

    print("Generating Gemini voiceover...")
    await _generate_gemini_tts_wav(VOICE_TEXT, voice)

    bgm = _get_background_track()

    scenes = [
        SceneClip(
            video_url=str(clip1),
            duration_seconds=5.6,
            caption_text="MOST INVESTORS CHASE STORIES",
            scene_number=1,
        ),
        SceneClip(
            video_url=str(clip2),
            duration_seconds=5.6,
            caption_text="MUNGER RULE: AVOID STUPID",
            scene_number=2,
        ),
        SceneClip(
            video_url=str(clip2),
            duration_seconds=5.6,
            caption_text="DROP A TICKER FOR A MOAT SCORE",
            scene_number=3,
        ),
    ]

    print("Rendering final video...")
    renderer = MoviePyRenderer(output_dir=out_dir)
    result = await renderer.render_composition(
        CreatomateRenderRequest(
            scenes=scenes,
            voiceover_url=str(voice),
            background_music_url=bgm,
            background_music_volume=0.22,
            width=1080,
            height=1920,
            fps=30,
        )
    )

    if not result.success or not result.output_path:
        raise RuntimeError(f"Render failed: {result.error_message}")

    final_path = out_dir / f"munger_talking_candle_{ts}.mp4"
    final_path.write_bytes(Path(result.output_path).read_bytes())
    print(f"Final video: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
