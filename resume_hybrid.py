"""Resume hybrid pipeline — generate clips 2 & 3, stitch all 3."""
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

OUT_DIR = Path(__file__).parent / "output" / "hybrid_20260226_114005"
KLING_MODEL = "fal-ai/kling-video/v3/pro/image-to-video"
CLIP_DURATION = "5"
ASPECT_RATIO = "9:16"


def get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


async def main():
    import fal_client
    import httpx

    # Load concept
    with open(OUT_DIR / "concept.json") as f:
        concept = json.load(f)

    scenes_to_gen = [s for s in concept["scenes"] if s["scene_number"] in (2, 3)]
    clips_dir = OUT_DIR / "clips"

    for scene in scenes_to_gen:
        num = scene["scene_number"]
        img_path = OUT_DIR / "images" / f"scene_{num:02d}.png"
        clip_path = clips_dir / f"clip_{num:02d}.mp4"

        if clip_path.exists():
            print(f"   Scene {num}: already exists, skipping")
            continue

        print(f"\n   Scene {num}/3: {scene['caption']}")
        print(f"   Uploading image to fal CDN...")
        image_url = await fal_client.upload_file_async(str(img_path))
        print(f"   [OK] Uploaded: {image_url[:80]}...")

        print(f"   Generating {CLIP_DURATION}s video with Kling 3.0 Pro...")
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
                print(f"   [FAIL] No video URL: {list(result.keys())}")
                continue

            print(f"   [OK] Downloading...")
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(video_url)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

            size_mb = clip_path.stat().st_size / 1024 / 1024
            print(f"   [OK] {size_mb:.1f}MB -> {clip_path.name}")

        except Exception as e:
            print(f"   [FAIL] {str(e)[:300]}")

    # Stitch all 3 clips
    print(f"\n{'='*60}")
    print("[RENDER] Stitching 3 clips with crossfades + text overlay")
    print(f"{'='*60}")

    clip_paths = sorted(clips_dir.glob("clip_*.mp4"))
    print(f"   Found {len(clip_paths)} clips: {[p.name for p in clip_paths]}")

    if len(clip_paths) < 2:
        print("   [ABORT] Need at least 2 clips")
        return

    ffmpeg = get_ffmpeg()
    crossfade = 0.5
    clip_dur = float(CLIP_DURATION)

    # Build xfade filter
    inputs = []
    for cp in clip_paths:
        inputs.extend(["-i", str(cp)])

    filter_parts = []
    offset = clip_dur - crossfade

    filter_parts.append(
        f"[0][1]xfade=transition=fade:duration={crossfade}:offset={offset}[v1]"
    )
    for i in range(2, len(clip_paths)):
        offset += clip_dur - crossfade
        filter_parts.append(
            f"[v{i-1}][{i}]xfade=transition=fade:duration={crossfade}:offset={offset}[v{i}]"
        )

    last = f"v{len(clip_paths)-1}"

    # Add hook text overlay (first 3 seconds)
    hook = concept["hook_text"].replace("'", "\\'").replace(":", "\\:")
    filter_parts.append(
        f"[{last}]drawtext=text='{hook}'"
        f":fontsize=64:fontcolor=white:borderw=3:bordercolor=black"
        f":x=(w-text_w)/2:y=h*0.08"
        f":enable='between(t,0,3)'[vout]"
    )

    filter_complex = ";".join(filter_parts)
    final_path = OUT_DIR / "final_video.mp4"

    cmd = [ffmpeg, "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(final_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   [WARN] Crossfade failed: {result.stderr[:500]}")
        # Fallback: simple concat
        print("   [FALLBACK] Simple concat...")
        concat_file = OUT_DIR / "concat.txt"
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{str(cp.absolute()).replace(chr(92), '/')}'\n")
        subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(final_path),
        ], check=True, capture_output=True)

    # Stats
    dur_result = subprocess.run([ffmpeg, "-i", str(final_path)], capture_output=True, text=True)
    for line in dur_result.stderr.split("\n"):
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            print(f"   Duration: {t}")
            break

    size_mb = final_path.stat().st_size / 1024 / 1024
    print(f"   Size: {size_mb:.1f}MB")
    print(f"   Output: {final_path}")
    print(f"\n   DONE!")


if __name__ == "__main__":
    asyncio.run(main())
