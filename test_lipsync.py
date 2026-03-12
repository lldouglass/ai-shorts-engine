"""Test lip-sync with fal.ai HeyGen Avatar IV Image-to-Video."""
import os
from dotenv import load_dotenv
load_dotenv()

import fal_client
import requests
import json
from pathlib import Path

OUT = Path("output/hook_test")

# First, upload our image and audio to fal storage
print("Uploading candle image to fal storage...")
image_url = fal_client.upload_file(str(OUT / "candle_character.png"))
print(f"Image URL: {image_url}")

print("Uploading hook audio to fal storage...")
audio_url = fal_client.upload_file(str(OUT / "hook_audio.mp3"))
print(f"Audio URL: {audio_url}")

# Try HeyGen Image to Video
print("\n--- Trying HeyGen Avatar IV Image-to-Video ---")
try:
    result = fal_client.subscribe(
        "fal-ai/heygen/avatar4/image-to-video",
        arguments={
            "image_url": image_url,
            "audio_url": audio_url,
        },
        with_logs=True,
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Download the video
    if "video" in result and "url" in result["video"]:
        video_url = result["video"]["url"]
    elif "video_url" in result:
        video_url = result["video_url"]
    else:
        video_url = None
        print(f"No video URL found in result keys: {list(result.keys())}")
    
    if video_url:
        video_data = requests.get(video_url).content
        out_path = OUT / "heygen_lipsync_test.mp4"
        out_path.write_bytes(video_data)
        print(f"Saved: {out_path} ({len(video_data)/1024:.0f} KB)")
        
except Exception as e:
    print(f"HeyGen error: {e}")
    print("\n--- Trying fal.ai SadTalker as fallback ---")
    try:
        result = fal_client.subscribe(
            "fal-ai/sadtalker",
            arguments={
                "source_image_url": image_url,
                "driven_audio_url": audio_url,
            },
            with_logs=True,
        )
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e2:
        print(f"SadTalker error: {e2}")
        
        # Try LivePortrait
        print("\n--- Trying fal.ai LivePortrait ---")
        try:
            result = fal_client.subscribe(
                "fal-ai/liveportrait",
                arguments={
                    "image_url": image_url,
                    "audio_url": audio_url,
                },
                with_logs=True,
            )
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e3:
            print(f"LivePortrait error: {e3}")

print("\nDone.")
