"""Generate bunny brainrot hook exactly from provided bunny image (3s + audible music).

Purpose: fix prior issues where bunny identity drifted and music was too quiet.

Pipeline:
1) Build 3s bunny dance intro directly from the exact image (no Veo identity drift)
2) Stitch with existing education clip
3) Gemini TTS voiceover
4) Stronger background music bed
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
from moviepy import ColorClip, CompositeVideoClip, ImageClip, concatenate_videoclips

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shorts_engine.adapters.renderer.creatomate import CreatomateRenderRequest, SceneClip
from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer

BUNNY_IMAGE = Path(
    r"C:\Users\Logan\.openclaw\media\inbound\file_147---51401cbf-1d66-45b4-80aa-d21b1082cb23.jpg"
)

# Reuse existing generated education clip to move fast.
EDU_CLIP = Path(r"C:\Users\Logan\Downloads\ai-shorts-engine\output\samples\bunny_edu_raw_20260305_155517.mp4")

# Louder/more rhythmic track than ambient_pad.
BGM_TRACK = Path(r"C:\Users\Logan\Downloads\ai-shorts-engine\storage\FEEL_IT_clip.mp3")

VOICE_TEXT = (
    "You just got caught brain rotting. Good. "
    "Now let us avoid one investing mistake Charlie Munger hated. "
    "Do not buy stories, buy businesses with pricing power. "
    "If customers stay even after price increases, that is a moat. "
    "If profits need perfect conditions to survive, skip it. "
    "Drop a ticker and I will score the moat."
)


def _build_bunny_dance_intro(out_path: Path) -> Path:
    """Create a 3-second 'dancing' intro from the exact bunny image."""
    if not BUNNY_IMAGE.exists():
        raise FileNotFoundError(f"Bunny image not found: {BUNNY_IMAGE}")

    W, H = 1080, 1920
    beat_duration = 0.25  # 12 beats = 3 seconds
    beats = 12

    # Color-shifting backgrounds to increase perceived motion/energy.
    bg_palette = [
        (26, 18, 45),
        (35, 10, 60),
        (18, 26, 65),
        (52, 14, 46),
    ]

    base = ImageClip(str(BUNNY_IMAGE)).resized(height=980)
    clips = []

    for i in range(beats):
        bg_color = bg_palette[i % len(bg_palette)]
        bg = ColorClip(size=(W, H), color=bg_color).with_duration(beat_duration)

        # Beat-based bounce + side-to-side wiggle + scale pulse.
        pulse = 1.08 if i % 2 == 0 else 0.96
        bunny = base.resized(height=int(980 * pulse)).with_duration(beat_duration)

        x_jitter = -58 if i % 2 == 0 else 58
        y_bounce = -20 if i % 4 in (0, 1) else 18

        x = int((W - bunny.w) / 2 + x_jitter)
        y = int(H * 0.46 + y_bounce)

        frame = CompositeVideoClip(
            [bg, bunny.with_position((x, y))],
            size=(W, H),
        ).with_duration(beat_duration)
        clips.append(frame)

    intro = concatenate_videoclips(clips, method="compose")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intro.write_videofile(
        str(out_path),
        fps=30,
        codec="libx264",
        audio=False,
        logger=None,
    )

    # Cleanup clip resources.
    intro.close()
    base.close()
    for c in clips:
        c.close()

    return out_path


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

    intro_path = out_dir / f"bunny_exact_intro_3s_{ts}.mp4"
    voice_path = out_dir / f"bunny_exact_voice_{ts}.wav"

    print("Building exact bunny 3s dance intro...")
    _build_bunny_dance_intro(intro_path)

    if not EDU_CLIP.exists():
        raise FileNotFoundError(f"Education clip missing: {EDU_CLIP}")

    print("Generating Gemini voiceover...")
    await _generate_gemini_tts_wav(VOICE_TEXT, voice_path)

    if not BGM_TRACK.exists():
        raise FileNotFoundError(f"BGM track missing: {BGM_TRACK}")

    scenes = [
        SceneClip(
            video_url=str(intro_path),
            duration_seconds=3.0,
            caption_text="YOU JUST GOT CAUGHT BRAIN ROTTING",
            scene_number=1,
        ),
        SceneClip(
            video_url=str(EDU_CLIP),
            duration_seconds=9.2,
            caption_text="BUY PRICING POWER, NOT HYPE",
            scene_number=2,
        ),
        SceneClip(
            video_url=str(EDU_CLIP),
            duration_seconds=4.3,
            caption_text="COMMENT A TICKER FOR A MOAT RATING",
            scene_number=3,
        ),
    ]

    print("Rendering final with louder music...")
    renderer = MoviePyRenderer(output_dir=out_dir)
    result = await renderer.render_composition(
        CreatomateRenderRequest(
            scenes=scenes,
            voiceover_url=str(voice_path),
            background_music_url=str(BGM_TRACK),
            background_music_volume=0.34,
            width=1080,
            height=1920,
            fps=30,
        )
    )

    if not result.success or not result.output_path:
        raise RuntimeError(f"Render failed: {result.error_message}")

    final_path = out_dir / f"bunny_brainrot_3s_music_fix_{ts}.mp4"
    final_path.write_bytes(Path(result.output_path).read_bytes())
    print(f"Final video: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
