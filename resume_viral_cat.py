"""Resume the viral cat video generation from scene 4.

Scenes 1-3 + voiceover already generated on Feb 23.
This script generates scenes 4-8, then renders the final video.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer
from shorts_engine.adapters.renderer.creatomate import CreatomateRenderRequest, SceneClip
from shorts_engine.adapters.video_gen.veo import VeoProvider
from shorts_engine.adapters.video_gen.base import VideoGenRequest
from shorts_engine.presets.styles import get_preset
from shorts_engine.utils.frame_extraction import extract_last_frame
from shorts_engine.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output" / "viral_cat"
CLIPS_DIR = OUTPUT_DIR / "clips"

# The original 8-scene plan from the first run
SCENES = [
    {"num": 1, "caption": "Meet Oliver, the chef!", "duration": 5.0,
     "prompt": "Close-up of an adorable orange tabby cat wearing bright yellow dish gloves, standing on hind legs at a kitchen counter next to a stove, holding a knife with determination, warm kitchen lighting, anime realism style"},
    {"num": 2, "caption": "Oops! Egg-cellent failure!", "duration": 5.0,
     "prompt": "An orange tabby cat opens a refrigerator door and eggs fall crashing to the floor, yolks splattering everywhere, the cat looks surprised but determined, warm kitchen, anime realism style"},
    {"num": 3, "caption": "Flour power unleashed!", "duration": 5.0,
     "prompt": "An orange tabby cat tugging at a bag of flour on a kitchen counter, white flour cloud exploding into the air covering everything, the kitchen looks like a winter wonderland, anime realism style"},
    {"num": 4, "caption": "The heat is on!", "duration": 5.0,
     "prompt": "An orange tabby cat on a kitchen countertop flipping on a stove burner, a pan clattering and bubbling with chaotic cooking, steam and small flames, dramatic kitchen lighting, anime realism style"},
    {"num": 5, "caption": "What happened here?", "duration": 5.0,
     "prompt": "A young woman opening her front door, jaw dropping in shock at the sight of a completely destroyed flour-covered kitchen, eggs on the floor, pots everywhere, dramatic reveal shot, anime realism style"},
    {"num": 6, "caption": "His true masterpiece!", "duration": 5.0,
     "prompt": "A flour-covered orange tabby cat sitting proudly on a messy kitchen counter, presenting a pristine untouched can of cat food placed carefully in a cereal bowl, looking up with proud innocent eyes, anime realism style"},
    {"num": 7, "caption": "Love in the chaos!", "duration": 5.0,
     "prompt": "A young woman laughing through tears of joy, kneeling down in a destroyed kitchen to pet a flour-covered orange tabby cat, warm golden light streaming through the window, emotional moment, anime realism style"},
    {"num": 8, "caption": "Dinner together at last!", "duration": 5.0,
     "prompt": "An orange tabby cat and a young woman sitting side by side on the kitchen floor surrounded by the mess, sharing a quiet moment, the cat eating from its bowl while she smiles at it, warm cozy lighting, anime realism style"},
]


async def check_existing_clips() -> list[int]:
    """Check which scene clips already exist and are valid (>100KB)."""
    existing = []
    for scene in SCENES:
        clip_path = CLIPS_DIR / f"scene_{scene['num']:02d}.mp4"
        if clip_path.exists() and clip_path.stat().st_size > 100_000:
            existing.append(scene["num"])
            print(f"   [EXISTS] Scene {scene['num']} ({clip_path.stat().st_size:,} bytes)")
        elif clip_path.exists():
            print(f"   [SMALL]  Scene {scene['num']} ({clip_path.stat().st_size:,} bytes) - will regenerate")
        else:
            print(f"   [MISSING] Scene {scene['num']}")
    return existing


async def generate_missing_clips(existing: list[int]) -> list[Path]:
    """Generate only the missing scene clips."""
    preset = get_preset("ANIME_REALISM")
    style_suffix = f", {preset.format_style_prompt()}" if preset else ""
    negative_prompt = preset.format_negative_prompt() if preset else None

    veo = VeoProvider()
    clip_paths = []

    # Get the last existing clip's frame for chaining
    previous_frame = None
    for scene_num in sorted(existing, reverse=True):
        clip_path = CLIPS_DIR / f"scene_{scene_num:02d}.mp4"
        if clip_path.exists():
            try:
                previous_frame = extract_last_frame(clip_path)
                print(f"\n   [CHAIN] Starting chain from scene {scene_num} last frame")
                break
            except Exception:
                pass

    for scene in SCENES:
        clip_path = CLIPS_DIR / f"scene_{scene['num']:02d}.mp4"

        # Skip existing valid clips
        if scene["num"] in existing:
            clip_paths.append(clip_path)
            # Update chain frame from this clip
            try:
                previous_frame = extract_last_frame(clip_path)
            except Exception:
                previous_frame = None
            continue

        print(f"\n   Scene {scene['num']}/8: {scene['caption']}")
        prompt = f"{scene['prompt']}{style_suffix}"

        request = VideoGenRequest(
            prompt=prompt,
            duration_seconds=int(scene["duration"]),
            aspect_ratio="9:16",
            negative_prompt=negative_prompt,
            reference_images=[previous_frame] if previous_frame else None,
        )

        result = await veo.generate(request)

        if not result.success:
            print(f"   [FAIL] {result.error_message}")
            previous_frame = None
            continue

        video_url = result.metadata.get("video_url") if result.metadata else None
        download_headers = result.metadata.get("download_headers", {}) if result.metadata else {}

        if video_url:
            import httpx
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                try:
                    response = await client.get(video_url, headers=download_headers)
                    response.raise_for_status()
                    clip_path.write_bytes(response.content)
                    clip_paths.append(clip_path)
                    print(f"   [OK] Saved ({len(response.content):,} bytes)")

                    try:
                        previous_frame = extract_last_frame(clip_path)
                        print(f"   [CHAIN] Extracted last frame")
                    except Exception as e:
                        print(f"   [WARN] Frame extraction failed: {e}")
                        previous_frame = None
                except Exception as e:
                    print(f"   [FAIL] Download: {e}")
                    previous_frame = None
        else:
            print("   [FAIL] No video URL")
            previous_frame = None

        # Rate limit pause
        if scene["num"] < 8:
            print("   [WAIT] Rate limit pause (20s)...")
            await asyncio.sleep(20)

    return clip_paths


async def render_final(clip_paths: list[Path]) -> Path:
    """Render final video from all clips + voiceover."""
    print("\n" + "=" * 60)
    print("[RENDER] Rendering final video...")
    print("=" * 60)

    voiceover_path = OUTPUT_DIR / "voiceover.mp3"
    if not voiceover_path.exists():
        print("[ERROR] voiceover.mp3 not found!")
        raise FileNotFoundError("voiceover.mp3 missing")

    renderer = MoviePyRenderer(output_dir=OUTPUT_DIR)

    scene_clips = []
    for i, clip_path in enumerate(clip_paths):
        scene_clips.append(
            SceneClip(
                video_url=str(clip_path.absolute()),
                duration_seconds=SCENES[i]["duration"],
                caption_text=None,
                scene_number=SCENES[i]["num"],
            )
        )

    print(f"   Clips: {len(scene_clips)}")
    print(f"   Voiceover: {voiceover_path}")

    request = CreatomateRenderRequest(
        scenes=scene_clips,
        voiceover_url=str(voiceover_path.absolute()),
        width=1080,
        height=1920,
        fps=30,
    )

    result = await renderer.render_composition(request)

    if not result.success:
        raise RuntimeError(f"Render failed: {result.error_message}")

    print(f"   [OK] Rendered to {result.output_path}")
    print(f"   Duration: {result.duration_seconds}s")
    return result.output_path


async def main():
    print("\n" + "=" * 60)
    print("  RESUME: Oliver's Kitchen Catastrophe")
    print("=" * 60)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Check what we have
    print("\n[CHECK] Existing clips:")
    existing = await check_existing_clips()
    missing = [s["num"] for s in SCENES if s["num"] not in existing]

    if not missing:
        print("\n[OK] All 8 scenes exist!")
    else:
        print(f"\n[GEN] Need to generate scenes: {missing}")
        clip_paths = await generate_missing_clips(existing)
        if len(clip_paths) < 8:
            print(f"\n[WARN] Only {len(clip_paths)}/8 clips available")
    
    # Collect all valid clips in order
    all_clips = []
    for scene in SCENES:
        clip_path = CLIPS_DIR / f"scene_{scene['num']:02d}.mp4"
        if clip_path.exists() and clip_path.stat().st_size > 100_000:
            all_clips.append(clip_path)

    if len(all_clips) < 3:
        print(f"\n[ERROR] Only {len(all_clips)} valid clips. Need at least 3.")
        return

    # Render
    final_path = await render_final(all_clips)

    print("\n" + "=" * 60)
    print("  VIDEO COMPLETE!")
    print("=" * 60)
    print(f"   Output: {final_path}")
    print(f"   Clips used: {len(all_clips)}/8")


if __name__ == "__main__":
    asyncio.run(main())
