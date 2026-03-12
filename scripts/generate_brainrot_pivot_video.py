"""Generate first Brainrot -> Education pivot short video.

Format:
1) Brainrot visual hook
2) Hard pivot line
3) Educational value drop
4) CTA

Uses current defaults:
- Veo for visuals
- Edge TTS narrator voice
- Timed subtitles from voice boundaries
- Default background music from settings
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shorts_engine.adapters.renderer.creatomate import CreatomateRenderRequest, SceneClip
from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer
from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.adapters.video_gen.veo import VeoProvider
from shorts_engine.adapters.voiceover.base import VoiceoverRequest
from shorts_engine.adapters.voiceover.edge_tts import EdgeTTSProvider
from shorts_engine.jobs.render_pipeline import (
    _build_timed_captions_from_word_boundaries,
    _resolve_background_music_defaults,
)

BRAINROT_PROMPT = (
    "Fast-paced viral style AI visual hook, quirky surreal transitions, bright color bursts, "
    "high-energy internet meme aesthetic, attention grabbing, vertical 9:16, cinematic quality"
)

EDU_PROMPT = (
    "Cinematic educational personal finance scene, person comparing used car costs on laptop, "
    "maintenance receipts and calculator on desk, clear charts, trustworthy modern look, "
    "high contrast, clean composition, vertical 9:16"
)

VOICE_LINE = (
    "You just got caught brain rotting. Good. "
    "Now use that attention to save real money. "
    "If you are buying a used car, do not start with sticker price. "
    "Start with five year maintenance cost. "
    "Cheap upfront can become expensive fast. "
    "Comment your car and I will tell you what to watch out for."
)


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


async def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    brainrot_clip = out_dir / f"brainrot_hook_raw_{ts}.mp4"
    edu_clip = out_dir / f"brainrot_edu_raw_{ts}.mp4"
    voice_path = out_dir / f"brainrot_voice_{ts}.mp3"
    final_path = out_dir / f"brainrot_pivot_first_video_{ts}.mp4"

    veo = VeoProvider()
    if not await veo.health_check():
        raise RuntimeError("Veo health check failed. Verify GOOGLE_API_KEY.")

    print("Generating brainrot hook clip...")
    await _generate_veo_clip(veo, BRAINROT_PROMPT, brainrot_clip)

    print("Generating education clip...")
    await _generate_veo_clip(veo, EDU_PROMPT, edu_clip)

    print("Generating voiceover...")
    tts = EdgeTTSProvider()
    voice_result = await tts.generate(
        VoiceoverRequest(
            text=VOICE_LINE,
            voice_id="narrator",
            speed=1.04,
        )
    )
    if not voice_result.success or not voice_result.audio_data:
        raise RuntimeError(f"Voiceover failed: {voice_result.error_message}")

    voice_path.write_bytes(voice_result.audio_data)
    word_boundaries = (voice_result.metadata or {}).get("word_boundaries", [])
    timed_captions = _build_timed_captions_from_word_boundaries(word_boundaries)

    bgm_url, bgm_volume = _resolve_background_music_defaults(None, None)

    # Hard cut structure: brainrot -> pivot -> education -> CTA
    scenes = [
        SceneClip(video_url=str(brainrot_clip), duration_seconds=2.2, scene_number=1),
        SceneClip(video_url=str(brainrot_clip), duration_seconds=1.0, scene_number=2),
        SceneClip(video_url=str(edu_clip), duration_seconds=7.8, scene_number=3),
        SceneClip(video_url=str(edu_clip), duration_seconds=3.2, scene_number=4),
    ]

    renderer = MoviePyRenderer(output_dir=out_dir)
    render_result = await renderer.render_composition(
        CreatomateRenderRequest(
            scenes=scenes,
            voiceover_url=str(voice_path),
            timed_captions=timed_captions,
            background_music_url=bgm_url,
            background_music_volume=bgm_volume,
            width=1080,
            height=1920,
            fps=30,
        )
    )

    if not render_result.success or not render_result.output_path:
        raise RuntimeError(f"Render failed: {render_result.error_message}")

    final_path.write_bytes(Path(render_result.output_path).read_bytes())
    print(f"Final video: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
