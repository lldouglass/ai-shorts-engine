"""Generate a quick hook sample using Gemini TTS voice (Google API).

This is a quality A/B sample against Edge TTS.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import wave
from datetime import datetime
from pathlib import Path
import sys

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shorts_engine.adapters.renderer.creatomate import CreatomateRenderRequest, SceneClip
from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer
from shorts_engine.jobs.render_pipeline import _resolve_background_music_defaults

TEXT = (
    "You just got caught brain rotting. Good. "
    "Now use that attention to learn one thing that saves you money."
)
VOICE_NAME = "Kore"


async def generate_gemini_tts_wav(text: str, out_wav: Path) -> Path:
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
                    "prebuiltVoiceConfig": {
                        "voiceName": VOICE_NAME,
                    }
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

    # Use latest generated hook clip if present.
    candidate_clips = sorted(out_dir.glob("brainrot_hook_raw_*.mp4"))
    if not candidate_clips:
        raise RuntimeError("No brainrot_hook_raw clip found. Generate one first.")
    clip_path = candidate_clips[-1]

    wav_path = out_dir / f"gemini_tts_hook_{ts}.wav"
    await generate_gemini_tts_wav(TEXT, wav_path)

    bgm_url, bgm_volume = _resolve_background_music_defaults(None, None)

    renderer = MoviePyRenderer(output_dir=out_dir)
    result = await renderer.render_composition(
        CreatomateRenderRequest(
            scenes=[
                SceneClip(
                    video_url=str(clip_path),
                    duration_seconds=6.0,
                    caption_text="YOU JUST GOT CAUGHT BRAIN ROTTING",
                    scene_number=1,
                )
            ],
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

    final_path = out_dir / f"gemini_tts_hook_sample_{ts}.mp4"
    final_path.write_bytes(Path(result.output_path).read_bytes())
    print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
