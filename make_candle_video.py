"""End-to-end talking candle video: topic → finished vertical video with captions."""
import os
import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

import fal_client
import requests
from pathlib import Path

OUT = Path("output/candle_videos")
OUT.mkdir(parents=True, exist_ok=True)


def generate_audio(text, output_name, voice="en-US-AndrewNeural", rate="-5%"):
    """Generate TTS audio with Edge TTS."""
    import edge_tts
    
    async def _gen():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        path = str(OUT / f"{output_name}_audio.mp3")
        await communicate.save(path)
        return path
    
    return asyncio.run(_gen())


def upload_and_lipsync(image_path_or_url, audio_path):
    """Upload assets and run HeyGen lip-sync via fal.ai."""
    # Upload audio
    print("Uploading audio...")
    audio_url = fal_client.upload_file(audio_path)
    print(f"Audio URL: {audio_url}")
    
    # Use image URL if already uploaded, otherwise upload
    if image_path_or_url.startswith("http"):
        image_url = image_path_or_url
    else:
        print("Uploading image...")
        image_url = fal_client.upload_file(image_path_or_url)
    print(f"Image URL: {image_url}")
    
    # Run HeyGen
    print("Running HeyGen lip-sync...")
    result = fal_client.subscribe(
        "fal-ai/heygen/avatar4/image-to-video",
        arguments={
            "image_url": image_url,
            "audio_url": audio_url,
        },
        with_logs=True,
    )
    
    video_url = result["video"]["url"]
    return video_url


def post_process_video(raw_video_path, caption_text, output_name):
    """Crop to vertical + add captions."""
    from post_process import post_process
    output_path = str(OUT / f"{output_name}_final.mp4")
    post_process(raw_video_path, caption_text, output_path=output_path)
    return output_path


def get_video_metadata(video_path: str) -> dict:
    """Extract delivery metadata for Distribution header."""
    from moviepy import VideoFileClip

    p = Path(video_path)
    size_bytes = p.stat().st_size
    clip = VideoFileClip(str(p))
    w, h = clip.size
    fps = float(clip.fps)
    duration = float(clip.duration)
    clip.close()

    bitrate_mbps = (size_bytes * 8.0 / max(duration, 0.001)) / 1_000_000.0

    return {
        "source_file": p.name,
        "width": int(w),
        "height": int(h),
        "fps": fps,
        "bitrate_mbps": bitrate_mbps,
        "duration": duration,
    }


def build_distribution_caption(
    *,
    brand: str,
    title: str,
    source_file: str,
    width: int,
    height: int,
    fps: float,
    bitrate_mbps: float,
    final_caption: str,
) -> str:
    """Build exact READY_FOR_DISTRIBUTION header format."""
    return (
        "READY_FOR_DISTRIBUTION\n"
        f"BRAND: {brand}\n"
        f"TITLE: {title}\n"
        f"SOURCE_FILE: {source_file}\n"
        f"RESOLUTION: {width}x{height}\n"
        f"FPS: {fps:.2f}\n"
        f"BITRATE_MBPS: {bitrate_mbps:.2f}\n"
        f"FINAL_CAPTION: {final_caption}"
    )


def make_video(script_text, output_name, image_url=None, brand="moatifi", title=None, final_caption=None):
    """Full pipeline: text → finished video + distribution metadata."""
    # Default candle character image (already on fal storage)
    if image_url is None:
        # Use the existing candle character
        local_image = "output/hook_test/candle_character.png"
        if os.path.exists(local_image):
            image_url = local_image
        else:
            raise FileNotFoundError("No candle character image found. Run test_hook.py first.")
    
    print(f"\n{'='*50}")
    print(f"Making video: {output_name}")
    print(f"Script: {script_text}")
    print(f"{'='*50}\n")
    
    # 1. Generate audio
    print("Step 1: Generating audio...")
    audio_path = generate_audio(script_text, output_name)
    print(f"Audio saved: {audio_path}")
    
    # 2. Lip-sync
    print("\nStep 2: Running lip-sync...")
    video_url = upload_and_lipsync(image_url, audio_path)
    
    # Download raw video
    raw_path = str(OUT / f"{output_name}_raw.mp4")
    video_data = requests.get(video_url).content
    Path(raw_path).write_bytes(video_data)
    print(f"Raw video: {raw_path} ({len(video_data)/1024:.0f} KB)")
    
    # 3. Post-process
    print("\nStep 3: Post-processing (crop + captions)...")
    final_path = post_process_video(raw_path, script_text, output_name)

    # 4. Run QA
    print("\nStep 4: Running QA checks...")
    from video_qa import run_qa
    qa_result = run_qa(final_path)
    if not qa_result["passed"]:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\nQA FAILED: {', '.join(failed)}")
        print("Video NOT sent to Distribution. Fix issues and retry.")
        return {"final_video_path": final_path, "qa_passed": False, "qa_result": qa_result}

    # 5. Build distribution metadata/header
    print("\nStep 5: Building distribution header...")
    if title is None:
        title = output_name.replace("_", " ").title()
    if final_caption is None:
        final_caption = script_text

    meta = get_video_metadata(final_path)
    distribution_header = build_distribution_caption(
        brand=brand,
        title=title,
        source_file=meta["source_file"],
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        bitrate_mbps=meta["bitrate_mbps"],
        final_caption=final_caption,
    )

    header_path = OUT / f"{output_name}_distribution_header.txt"
    header_path.write_text(distribution_header, encoding="utf-8")

    print(f"\nFinal video: {final_path}")
    print(f"Header file: {header_path}")
    print("\n--- READY_FOR_DISTRIBUTION ---")
    print(distribution_header)

    return {
        "final_video_path": final_path,
        "distribution_header": distribution_header,
        "distribution_header_path": str(header_path),
        "metadata": meta,
        "qa_passed": True,
    }


if __name__ == "__main__":
    script = sys.argv[1] if len(sys.argv) > 1 else "The stock market is a device for transferring money from the impatient to the patient."
    name = sys.argv[2] if len(sys.argv) > 2 else "candle_video"
    make_video(script, name)
