"""Generate one Veo hook using the current default subtitle+BGM stack.

Uses:
- Veo clip generation
- Edge TTS voiceover (with word boundaries)
- Timed subtitle chunking from voice boundaries
- Default background music resolution from settings

Usage:
  python scripts/generate_veo_hook_default_stack.py
  python scripts/generate_veo_hook_default_stack.py "visual prompt" "voice line"
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

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
from shorts_engine.config import settings
from shorts_engine.jobs.render_pipeline import (
    _build_timed_captions_from_word_boundaries,
    _resolve_background_music_defaults,
)


DEFAULT_VISUAL_PROMPT = (
    "Vertical cinematic finance hook shot, dramatic push-in over a modern trading desk, "
    "high contrast shadows, glowing data overlays, premium ad aesthetic, dynamic depth"
)
DEFAULT_VOICE_LINE = "Most investors look at price first, but moat quality matters more."


async def main() -> int:
    visual_prompt = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_VISUAL_PROMPT
    voice_line = sys.argv[2].strip() if len(sys.argv) > 2 else DEFAULT_VOICE_LINE

    out_dir = ROOT / "output" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Generating Veo hook clip...")
    print(f"Model: {settings.veo_model}")

    veo = VeoProvider(model=settings.veo_model)
    if not await veo.health_check():
        print("Veo health check failed. Verify GOOGLE_API_KEY.")
        return 1

    video_req = VideoGenRequest(
        prompt=visual_prompt,
        duration_seconds=6,
        aspect_ratio="9:16",
    )

    video_result = await veo.generate(video_req)
    if not video_result.success:
        print(f"Veo generation failed: {video_result.error_message}")
        return 1

    video_url = (video_result.metadata or {}).get("video_url")
    download_headers = (video_result.metadata or {}).get("download_headers", {})
    if not video_url:
        print("No video_url returned from Veo")
        return 1

    raw_clip_path = out_dir / f"hook_default_raw_{ts}.mp4"
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        r = await client.get(video_url, headers=download_headers)
        r.raise_for_status()
        raw_clip_path.write_bytes(r.content)
    print(f"Saved raw clip: {raw_clip_path}")

    print("Generating Edge TTS voiceover...")
    tts = EdgeTTSProvider()
    voice_result = await tts.generate(
        VoiceoverRequest(
            text=voice_line,
            voice_id="narrator",
            speed=1.04,
        )
    )
    if not voice_result.success or not voice_result.audio_data:
        print(f"Edge TTS failed: {voice_result.error_message}")
        return 1

    voice_path = out_dir / f"hook_default_voice_{ts}.mp3"
    voice_path.write_bytes(voice_result.audio_data)
    print(f"Saved voiceover: {voice_path}")

    word_boundaries = (voice_result.metadata or {}).get("word_boundaries", [])
    timed_captions = _build_timed_captions_from_word_boundaries(word_boundaries)
    print(f"Timed subtitle chunks: {len(timed_captions)}")

    bgm_url, bgm_volume = _resolve_background_music_defaults(None, None)
    print(f"Background music: {bgm_url or 'none'} @ volume {bgm_volume}")

    renderer = MoviePyRenderer(output_dir=out_dir)
    render_result = await renderer.render_composition(
        CreatomateRenderRequest(
            scenes=[
                SceneClip(
                    video_url=str(raw_clip_path),
                    duration_seconds=6.0,
                    scene_number=1,
                )
            ],
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
        print(f"Render failed: {render_result.error_message}")
        return 1

    final_path = out_dir / f"hook_default_stack_{ts}.mp4"
    final_path.write_bytes(Path(render_result.output_path).read_bytes())

    print(f"Final sample: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
