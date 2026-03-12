"""Range Rover video using the proven candle pipeline: Nano Banana image + Edge TTS + HeyGen lip-sync."""
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

import fal_client
import requests
import edge_tts
from pathlib import Path
from post_process import post_process
from video_qa import run_qa

OUT = Path("output/car_videos")
OUT.mkdir(parents=True, exist_ok=True)

# Use the Nano Banana Range Rover image that was already generated
IMAGE_PATH = OUT / "range_rover_nb_front.jpg"

SCRIPT = (
    "I cost a hundred thousand dollars and I scored a two out of five on reliability. "
    "My air suspension fails, my electronics glitch, and I have been recalled five times. "
    "But hey, at least I look good in the shop."
)

VOICE = "en-US-AndrewNeural"
RATE = "-5%"  # Same natural rate as candle videos
OUTPUT_NAME = "range_rover_heygen"


def main():
    assert IMAGE_PATH.exists(), f"Missing image: {IMAGE_PATH}"
    print(f"Image: {IMAGE_PATH} ({IMAGE_PATH.stat().st_size/1024:.0f} KB)")

    # 1. Generate TTS audio (same settings as candle pipeline)
    print(f"Generating audio ({VOICE}, rate={RATE})...")
    audio_path = str(OUT / f"{OUTPUT_NAME}_audio.mp3")

    async def _tts():
        comm = edge_tts.Communicate(SCRIPT, VOICE, rate=RATE)
        await comm.save(audio_path)
    asyncio.run(_tts())
    print(f"Audio: {audio_path}")

    # 2. Upload image + audio to fal, run HeyGen lip-sync
    print("Uploading image...")
    image_url = fal_client.upload_file(str(IMAGE_PATH))
    print(f"Image URL: {image_url}")

    print("Uploading audio...")
    audio_url = fal_client.upload_file(audio_path)
    print(f"Audio URL: {audio_url}")

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
    raw_path = str(OUT / f"{OUTPUT_NAME}_raw.mp4")
    video_data = requests.get(video_url).content
    Path(raw_path).write_bytes(video_data)
    print(f"Raw video: {raw_path} ({len(video_data)/1024:.0f} KB)")

    # 3. Post-process (crop to 9:16 + upscale to 1080x1920 + captions)
    final_path = str(OUT / f"{OUTPUT_NAME}_final.mp4")
    post_process(raw_path, SCRIPT, output_path=final_path)

    # 4. QA
    print("\nRunning QA...")
    qa_result = run_qa(final_path)
    if not qa_result["passed"]:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\nQA FAILED: {', '.join(failed)}")
    else:
        print("\n[PASS] Ready for Distribution")

    return final_path


if __name__ == "__main__":
    main()
