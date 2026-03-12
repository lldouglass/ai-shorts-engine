"""Hybrid video pipeline: Imagen 4 Ultra -> Kling 3.0 Pro I2V.

Generates high-quality still images with Google's Imagen 4 Ultra,
then animates each with Kling 3.0 Pro image-to-video via fal.ai.
Final render stitches clips with ffmpeg + text overlays.

Cost per 15s video (3 scenes x 5s):
  - Imagen 4 Ultra: 3 x ~$0.06 = ~$0.18
  - Kling 3.0 Pro I2V: 15s x $0.224 = ~$3.36
  - Total: ~$3.54
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from google import genai
from google.genai import types as genai_types

# ── Config ──────────────────────────────────────────────────────────
IMAGE_MODEL = "imagen-4.0-ultra-generate-001"
KLING_MODEL = "fal-ai/kling-video/v3/pro/image-to-video"
CLIP_DURATION = "5"  # seconds per clip (string for fal API)
ASPECT_RATIO = "9:16"
OUTPUT_DIR = Path(__file__).parent / "output"
WIDTH, HEIGHT = 1080, 1920


def get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── Scene Definition ────────────────────────────────────────────────
def get_scenes() -> dict:
    """Return the video concept with image + motion prompts per scene."""
    return {
        "title": "What If Samurai Had Lightsabers",
        "hashtags": ["whatif", "samurai", "lightsaber", "ai", "alternatehistory", "scifi"],
        "hook_text": "What if samurai had lightsabers?",
        "scenes": [
            {
                "scene_number": 1,
                "image_prompt": (
                    "A lone samurai warrior in traditional dark armor standing on a hilltop at golden hour sunset, "
                    "drawing a glowing neon blue lightsaber katana from its sheath, the blade casting blue light "
                    "across his face and armor, wind blowing his hair, dramatic silhouette against orange and "
                    "purple sky, cinematic composition, photorealistic, vertical portrait 9:16 aspect ratio, "
                    "hyper detailed, 8K quality"
                ),
                "motion_prompt": (
                    "The samurai slowly draws the glowing lightsaber from its sheath, the blade igniting with "
                    "a bright blue glow. Wind ripples through his clothing and hair. The sunset sky shifts colors "
                    "behind him. Camera slowly pushes in on the warrior's determined face."
                ),
                "caption": "THE DRAW",
            },
            {
                "scene_number": 2,
                "image_prompt": (
                    "Two samurai warriors locked in an intense duel on a moonlit stone courtyard, one wielding "
                    "a glowing red lightsaber and the other a glowing blue lightsaber, their blades clashing with "
                    "bright sparks and energy crackling at the point of contact, cherry blossom petals frozen "
                    "mid-air around them, dramatic low-angle shot, cinematic lighting, photorealistic, "
                    "vertical portrait 9:16 aspect ratio, hyper detailed, 8K quality"
                ),
                "motion_prompt": (
                    "Two samurai clash their lightsabers together with explosive sparks flying. The red and blue "
                    "blades crackle with energy. Cherry blossom petals swirl violently around them from the force. "
                    "Both warriors push against each other with intense effort. Camera orbits slowly around the duel."
                ),
                "caption": "THE CLASH",
            },
            {
                "scene_number": 3,
                "image_prompt": (
                    "Aftermath of an epic battle, a single samurai standing victorious in a scorched stone courtyard, "
                    "lightsaber held at his side glowing faintly blue, hundreds of cherry blossom petals falling "
                    "gently like pink snow all around him, scorch marks and glowing embers on the ground, "
                    "ethereal moonlight breaking through clouds above, peaceful yet powerful atmosphere, "
                    "cinematic wide shot, photorealistic, vertical portrait 9:16 aspect ratio, hyper detailed, 8K quality"
                ),
                "motion_prompt": (
                    "The victorious samurai stands still as cherry blossom petals fall gently all around him like "
                    "pink snow. His lightsaber dims slowly. Embers glow and float upward from the scorched ground. "
                    "Moonlight shifts through parting clouds. The warrior slowly sheathes his blade. Serene atmosphere."
                ),
                "caption": "THE AFTERMATH",
            },
        ],
    }


# ── Step 1: Image Generation (Imagen 4 Ultra via Google) ───────────
def generate_images(scenes: list[dict], output_dir: Path) -> list[Path]:
    """Generate images with Imagen 4 Ultra."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for scene in scenes:
        num = scene["scene_number"]
        prompt = scene["image_prompt"]
        print(f"   Scene {num}/{len(scenes)}: {scene['caption']}")

        for attempt in range(3):
            try:
                response = client.models.generate_images(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    config=genai_types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="9:16",
                    ),
                )
                img_bytes = response.generated_images[0].image.image_bytes
                img_path = images_dir / f"scene_{num:02d}.png"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"   [OK] {len(img_bytes):,} bytes -> {img_path.name}")
                paths.append(img_path)
                break
            except Exception as e:
                err = str(e)[:200]
                print(f"   [FAIL] Attempt {attempt + 1}: {err}")
                if "RESOURCE_EXHAUSTED" in str(e):
                    print("   [WAIT] Rate limited, waiting 30s...")
                    time.sleep(30)
                elif attempt < 2:
                    time.sleep(5)

        # Rate limit pause between images
        if num < len(scenes):
            time.sleep(3)

    return paths


# ── Step 2: Animate with Kling 3.0 Pro I2V (fal.ai) ────────────────
async def animate_scenes(
    scenes: list[dict], image_paths: list[Path], output_dir: Path
) -> list[Path]:
    """Animate each image with Kling 3.0 Pro I2V via fal.ai."""
    import fal_client
    import httpx

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    for scene, img_path in zip(scenes, image_paths):
        num = scene["scene_number"]
        print(f"   Scene {num}/{len(scenes)}: Uploading image to fal CDN...")

        # Upload image to fal CDN
        image_url = await fal_client.upload_file_async(str(img_path))
        print(f"   [OK] Uploaded: {image_url[:80]}...")

        print(f"   Scene {num}/{len(scenes)}: Generating {CLIP_DURATION}s video with Kling 3.0 Pro...")
        try:
            result = await fal_client.subscribe_async(
                KLING_MODEL,
                arguments={
                    "prompt": scene["motion_prompt"],
                    "start_image_url": image_url,
                    "duration": CLIP_DURATION,
                    "aspect_ratio": ASPECT_RATIO,
                    "negative_prompt": "blur, distort, low quality, morphing, deformation",
                    "generate_audio": False,
                },
            )

            video_url = result.get("video", {}).get("url")
            if not video_url:
                print(f"   [FAIL] No video URL in response: {list(result.keys())}")
                continue

            print(f"   [OK] Video generated, downloading...")

            # Download the video
            clip_path = clips_dir / f"clip_{num:02d}.mp4"
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(video_url)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

            size_mb = clip_path.stat().st_size / 1024 / 1024
            print(f"   [OK] {size_mb:.1f}MB -> {clip_path.name}")
            clip_paths.append(clip_path)

        except Exception as e:
            print(f"   [FAIL] Kling generation error: {str(e)[:300]}")
            continue

    return clip_paths


# ── Step 3: Stitch clips with ffmpeg + text overlay ─────────────────
def render_final(
    clip_paths: list[Path], concept: dict, output_dir: Path
) -> tuple[Path, float, int]:
    """Stitch video clips with crossfades and text overlay."""
    ffmpeg = get_ffmpeg()
    final_path = output_dir / "final_video.mp4"

    if len(clip_paths) == 1:
        # Single clip, just add text overlay
        add_text_overlay(ffmpeg, clip_paths[0], final_path, concept["hook_text"])
    elif len(clip_paths) == 0:
        print("   [ERROR] No clips to render!")
        return final_path, 0, 0
    else:
        # Multiple clips: crossfade transitions
        crossfade = 0.5  # half second crossfade
        clip_duration = float(CLIP_DURATION)

        # Build xfade filter chain
        inputs = []
        for cp in clip_paths:
            inputs.extend(["-i", str(cp)])

        filter_parts = []
        offset = clip_duration - crossfade

        # First transition
        filter_parts.append(
            f"[0][1]xfade=transition=fade:duration={crossfade}:offset={offset}[v1]"
        )

        for i in range(2, len(clip_paths)):
            offset += clip_duration - crossfade
            prev = f"v{i - 1}"
            out = f"v{i}"
            filter_parts.append(
                f"[{prev}][{i}]xfade=transition=fade:duration={crossfade}:offset={offset}[{out}]"
            )

        last_label = f"v{len(clip_paths) - 1}"

        # Add text overlay to the final composited video
        # Hook text at top, scene captions at bottom
        hook = concept["hook_text"].replace("'", "\\'").replace(":", "\\:")
        filter_parts.append(
            f"[{last_label}]drawtext=text='{hook}'"
            f":fontsize=64:fontcolor=white:borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=h*0.08"
            f":enable='between(t,0,3)'[vout]"
        )

        filter_complex = ";".join(filter_parts)

        temp_path = output_dir / "temp_stitched.mp4"
        cmd = [ffmpeg, "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(temp_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   [WARN] Crossfade failed: {result.stderr[:300]}")
            print(f"   [FALLBACK] Simple concat...")
            # Fallback: simple concat
            concat_file = output_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for cp in clip_paths:
                    f.write(f"file '{str(cp.absolute()).replace(chr(92), '/')}'\n")
            subprocess.run([
                ffmpeg, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(temp_path),
            ], check=True, capture_output=True)

        # Add text overlay on fallback too
        if temp_path.exists():
            add_text_overlay(ffmpeg, temp_path, final_path, concept["hook_text"])
            temp_path.unlink(missing_ok=True)
        else:
            final_path = temp_path

    # Get final stats
    dur = get_duration(ffmpeg, final_path) if final_path.exists() else 0
    size = final_path.stat().st_size if final_path.exists() else 0
    return final_path, dur, size


def add_text_overlay(ffmpeg: str, input_path: Path, output_path: Path, text: str):
    """Add hook text overlay to video."""
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    filter_str = (
        f"drawtext=text='{safe_text}'"
        f":fontsize=64:fontcolor=white:borderw=3:bordercolor=black"
        f":x=(w-text_w)/2:y=h*0.08"
        f":enable='between(t,0,3)'"
    )
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], check=True, capture_output=True)


def get_duration(ffmpeg: str, path: Path) -> float:
    """Get media file duration via ffprobe."""
    result = subprocess.run(
        [ffmpeg, "-i", str(path)],
        capture_output=True, text=True,
    )
    for line in result.stderr.split("\n"):
        if "Duration:" in line:
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            parts = time_str.split(":")
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


# ── Main Pipeline ─────────────────────────────────────────────────
async def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"hybrid_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    concept = get_scenes()

    print("\n" + "=" * 60)
    print("  HYBRID PIPELINE: Imagen 4 Ultra -> Kling 3.0 Pro")
    print("=" * 60)
    print(f"  Title:  {concept['title']}")
    print(f"  Scenes: {len(concept['scenes'])} x {CLIP_DURATION}s")
    print(f"  Est. cost: ~$3.54")
    print(f"  Output: {out_dir}")

    # Save concept
    with open(out_dir / "concept.json", "w") as f:
        json.dump(concept, f, indent=2)

    # Step 1: Generate images
    print(f"\n{'=' * 60}")
    print(f"[1/3] IMAGES - Imagen 4 Ultra (Google)")
    print(f"{'=' * 60}")
    image_paths = generate_images(concept["scenes"], out_dir)
    print(f"\n   Generated: {len(image_paths)}/{len(concept['scenes'])} images")

    if len(image_paths) < len(concept["scenes"]):
        print(f"   [WARN] Missing {len(concept['scenes']) - len(image_paths)} images")
        if len(image_paths) == 0:
            print("   [ABORT] No images generated. Check GOOGLE_API_KEY and quota.")
            return

    # Step 2: Animate with Kling 3.0 Pro
    print(f"\n{'=' * 60}")
    print(f"[2/3] VIDEO - Kling 3.0 Pro I2V (fal.ai)")
    print(f"{'=' * 60}")
    clip_paths = await animate_scenes(concept["scenes"], image_paths, out_dir)
    print(f"\n   Generated: {len(clip_paths)}/{len(image_paths)} video clips")

    if len(clip_paths) == 0:
        print("   [ABORT] No clips generated. Check FAL_KEY and credits.")
        return

    # Step 3: Stitch + text overlay
    print(f"\n{'=' * 60}")
    print(f"[3/3] RENDER - Stitch + Text Overlay")
    print(f"{'=' * 60}")
    final_path, duration, size = render_final(clip_paths, concept, out_dir)

    print(f"\n{'=' * 60}")
    print("  VIDEO COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Output:   {final_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Size:     {size / 1024 / 1024:.1f}MB")
    print(f"  Title:    {concept['title']}")
    print(f"  Hashtags: {' '.join('#' + h for h in concept['hashtags'])}")
    print(f"\n  Images:   {out_dir / 'images'}")
    print(f"  Clips:    {out_dir / 'clips'}")


if __name__ == "__main__":
    asyncio.run(main())
