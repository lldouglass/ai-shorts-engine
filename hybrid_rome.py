"""Hybrid pipeline: Ancient Rome with Neon Lights.

Atmospheric beauty shots — plays to I2V strengths:
slow camera, environmental effects, single focus subjects.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from google import genai
from google.genai import types as genai_types

IMAGE_MODEL = "imagen-4.0-ultra-generate-001"
KLING_MODEL = "fal-ai/kling-video/v3/pro/image-to-video"
CLIP_DURATION = "5"
ASPECT_RATIO = "9:16"
OUTPUT_DIR = Path(__file__).parent / "output"


def get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


CONCEPT = {
    "title": "What If Ancient Rome Had Neon Lights",
    "hook_text": "What if Ancient Rome had neon lights?",
    "hashtags": ["whatif", "ancientrome", "neon", "cyberpunk", "ai", "alternatehistory"],
    "scenes": [
        {
            "scene_number": 1,
            "image_prompt": (
                "The Roman Colosseum at night, but covered in glowing neon signs written in Latin, "
                "vibrant pink and electric blue neon tubes lining the arches, wet stone streets reflecting "
                "all the neon colors like a mirror, light rain falling, puddles on ancient cobblestones "
                "reflecting the glow, atmospheric fog, cinematic wide shot looking up at the massive "
                "structure, photorealistic, hyper detailed, vertical portrait 9:16 aspect ratio, 8K"
            ),
            "motion_prompt": (
                "Light rain falls on wet cobblestone streets reflecting vibrant neon colors. "
                "Fog drifts slowly through the Colosseum arches. Neon signs flicker and pulse gently. "
                "Raindrops create ripples in the glowing puddles. Camera slowly tilts upward revealing "
                "the full neon-lit Colosseum. Atmospheric, cinematic."
            ),
            "caption": "THE COLOSSEUM",
        },
        {
            "scene_number": 2,
            "image_prompt": (
                "A lone Roman centurion in polished bronze armor walking down a narrow ancient Roman "
                "street at night, neon signs in Latin glowing on both sides of the street in warm orange "
                "and cool purple, his red cape flowing behind him, neon light reflecting off his armor, "
                "steam rising from street grates, dramatic backlighting, cinematic medium shot from behind, "
                "photorealistic, hyper detailed, vertical portrait 9:16 aspect ratio, 8K"
            ),
            "motion_prompt": (
                "The centurion walks slowly forward down the neon-lit street, his red cape flowing and "
                "swaying gently behind him. Steam rises from grates in the street. Neon signs cast shifting "
                "colored light across his polished armor. His footsteps echo. Camera follows from behind at "
                "a steady pace. Atmospheric, slow, cinematic."
            ),
            "caption": "THE CENTURION",
        },
        {
            "scene_number": 3,
            "image_prompt": (
                "Breathtaking wide aerial view of ancient Rome at night completely transformed with neon lights, "
                "every temple and building outlined in glowing neon, the Tiber river reflecting thousands of "
                "colorful neon lights like a river of liquid color, the sky has a faint aurora-like glow, "
                "tiny figures visible on bridges and streets below, epic scale, cyberpunk meets ancient world, "
                "cinematic aerial establishing shot, photorealistic, hyper detailed, vertical portrait 9:16 "
                "aspect ratio, 8K"
            ),
            "motion_prompt": (
                "Slow aerial pan across the neon-lit ancient Rome skyline. Thousands of neon lights shimmer "
                "and reflect in the Tiber river below. Faint aurora shifts colors in the sky above. "
                "Tiny boats drift on the glowing river. The camera glides smoothly forward over the city, "
                "revealing more neon-lit temples and buildings. Grand, epic, serene."
            ),
            "caption": "THE ETERNAL CITY",
        },
    ],
}


def generate_images(scenes, output_dir):
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for scene in scenes:
        num = scene["scene_number"]
        print(f"   Scene {num}/{len(scenes)}: {scene['caption']}")
        for attempt in range(3):
            try:
                response = client.models.generate_images(
                    model=IMAGE_MODEL,
                    prompt=scene["image_prompt"],
                    config=genai_types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="9:16",
                    ),
                )
                img_bytes = response.generated_images[0].image.image_bytes
                img_path = images_dir / f"scene_{num:02d}.png"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"   [OK] {len(img_bytes):,} bytes")
                paths.append(img_path)
                break
            except Exception as e:
                print(f"   [FAIL] Attempt {attempt+1}: {str(e)[:200]}")
                if "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(30)
                elif attempt < 2:
                    time.sleep(5)
        if num < len(scenes):
            time.sleep(3)
    return paths


async def animate_scenes(scenes, image_paths, output_dir):
    import fal_client
    import httpx

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    for scene, img_path in zip(scenes, image_paths):
        num = scene["scene_number"]
        print(f"\n   Scene {num}/{len(scenes)}: {scene['caption']}")
        print(f"   Uploading to fal CDN...")
        image_url = await fal_client.upload_file_async(str(img_path))
        print(f"   [OK] Uploaded")

        print(f"   Generating {CLIP_DURATION}s with Kling 3.0 Pro...")
        try:
            result = await fal_client.subscribe_async(
                KLING_MODEL,
                arguments={
                    "prompt": scene["motion_prompt"],
                    "start_image_url": image_url,
                    "duration": CLIP_DURATION,
                    "aspect_ratio": ASPECT_RATIO,
                    "negative_prompt": "blur, distort, low quality, morphing, deformation, fast motion",
                    "generate_audio": False,
                },
            )
            video_url = result.get("video", {}).get("url")
            if not video_url:
                print(f"   [FAIL] No video URL")
                continue

            clip_path = clips_dir / f"clip_{num:02d}.mp4"
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(video_url)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)
            print(f"   [OK] {clip_path.stat().st_size / 1024 / 1024:.1f}MB")
            clip_paths.append(clip_path)
        except Exception as e:
            print(f"   [FAIL] {str(e)[:300]}")
    return clip_paths


def stitch(clip_paths, concept, output_dir):
    ffmpeg = get_ffmpeg()
    final_path = output_dir / "final_video.mp4"
    crossfade = 0.5
    clip_dur = float(CLIP_DURATION)

    inputs = []
    for cp in clip_paths:
        inputs.extend(["-i", str(cp)])

    filter_parts = []
    offset = clip_dur - crossfade
    filter_parts.append(f"[0][1]xfade=transition=fade:duration={crossfade}:offset={offset}[v1]")
    for i in range(2, len(clip_paths)):
        offset += clip_dur - crossfade
        filter_parts.append(f"[v{i-1}][{i}]xfade=transition=fade:duration={crossfade}:offset={offset}[v{i}]")

    last = f"v{len(clip_paths)-1}"
    hook = concept["hook_text"].replace("'", "\\'").replace(":", "\\:")
    filter_parts.append(
        f"[{last}]drawtext=text='{hook}'"
        f":fontsize=64:fontcolor=white:borderw=3:bordercolor=black"
        f":x=(w-text_w)/2:y=h*0.08"
        f":enable='between(t,0,3)'[vout]"
    )

    cmd = [ffmpeg, "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(final_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   [WARN] Crossfade failed, simple concat...")
        concat_file = output_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{str(cp.absolute()).replace(chr(92), '/')}'\n")
        subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(final_path),
        ], check=True, capture_output=True)

    # Duration
    r = subprocess.run([ffmpeg, "-i", str(final_path)], capture_output=True, text=True)
    for line in r.stderr.split("\n"):
        if "Duration:" in line:
            print(f"   Duration: {line.split('Duration:')[1].split(',')[0].strip()}")
            break
    print(f"   Size: {final_path.stat().st_size / 1024 / 1024:.1f}MB")
    print(f"   Output: {final_path}")
    return final_path


async def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"hybrid_rome_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  HYBRID: Ancient Rome + Neon Lights")
    print("=" * 60)
    print(f"  Scenes: {len(CONCEPT['scenes'])} x {CLIP_DURATION}s (atmospheric)")
    print(f"  Output: {out_dir}")

    with open(out_dir / "concept.json", "w") as f:
        json.dump(CONCEPT, f, indent=2)

    print(f"\n[1/3] IMAGES — Imagen 4 Ultra")
    images = generate_images(CONCEPT["scenes"], out_dir)
    print(f"   {len(images)}/3 images done")

    if not images:
        print("   ABORT"); return

    print(f"\n[2/3] VIDEO — Kling 3.0 Pro I2V")
    clips = await animate_scenes(CONCEPT["scenes"], images, out_dir)
    print(f"\n   {len(clips)}/3 clips done")

    if len(clips) < 2:
        print("   ABORT — need at least 2 clips"); return

    print(f"\n[3/3] RENDER")
    stitch(clips, CONCEPT, out_dir)
    print("\n   DONE!")


if __name__ == "__main__":
    asyncio.run(main())
