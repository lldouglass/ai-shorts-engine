"""Resume Rome pipeline — generate clip 3, stitch all 3."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

OUT_DIR = Path(__file__).parent / "output" / "hybrid_rome_20260226_141020"
KLING_MODEL = "fal-ai/kling-video/v3/pro/image-to-video"
CLIP_DURATION = "5"
ASPECT_RATIO = "9:16"


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


async def main():
    import fal_client
    import httpx

    with open(OUT_DIR / "concept.json") as f:
        concept = json.load(f)

    scene = concept["scenes"][2]  # Scene 3
    img_path = OUT_DIR / "images" / "scene_03.png"
    clip_path = OUT_DIR / "clips" / "clip_03.mp4"

    if not clip_path.exists():
        print(f"   Scene 3: THE ETERNAL CITY")
        print(f"   Uploading to fal CDN...")
        image_url = await fal_client.upload_file_async(str(img_path))
        print(f"   [OK] Uploaded")

        print(f"   Generating 5s with Kling 3.0 Pro...")
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
            print(f"   [FAIL] No video URL"); return

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            with open(clip_path, "wb") as f:
                f.write(resp.content)
        print(f"   [OK] {clip_path.stat().st_size / 1024 / 1024:.1f}MB")
    else:
        print(f"   Scene 3 already exists")

    # Stitch all 3
    print(f"\n   Stitching 3 clips...")
    ffmpeg = get_ffmpeg()
    clip_paths = sorted((OUT_DIR / "clips").glob("clip_*.mp4"))
    print(f"   Clips: {[p.name for p in clip_paths]}")

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

    final_path = OUT_DIR / "final_video.mp4"
    cmd = [ffmpeg, "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(final_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   Crossfade failed, simple concat...")
        concat_file = OUT_DIR / "concat.txt"
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{str(cp.absolute()).replace(chr(92), '/')}'\n")
        subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(final_path),
        ], check=True, capture_output=True)

    r = subprocess.run([ffmpeg, "-i", str(final_path)], capture_output=True, text=True)
    for line in r.stderr.split("\n"):
        if "Duration:" in line:
            print(f"   Duration: {line.split('Duration:')[1].split(',')[0].strip()}")
            break
    print(f"   Size: {final_path.stat().st_size / 1024 / 1024:.1f}MB")
    print(f"   DONE!")

if __name__ == "__main__":
    asyncio.run(main())
