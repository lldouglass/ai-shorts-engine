"""Test vertical (9:16) output from fal.ai HeyGen."""
import os
from dotenv import load_dotenv
load_dotenv()

import fal_client
import requests
import json
from pathlib import Path

OUT = Path("output/hook_test")

# Reuse the already-uploaded assets from earlier
image_url = "https://v3b.fal.media/files/b/0a91276b/34AZxFmJ_qgzdAl7CeS46_candle_character.png"
audio_url = "https://v3b.fal.media/files/b/0a91276b/XEsQnUeI5MTTU4Cg8DoxH_hook_audio.mp3"

print("Generating vertical (9:16) lip-sync video...")
try:
    result = fal_client.subscribe(
        "fal-ai/heygen/avatar4/image-to-video",
        arguments={
            "image_url": image_url,
            "audio_url": audio_url,
            "aspect_ratio": "9:16",
        },
        with_logs=True,
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if "video" in result and "url" in result["video"]:
        video_url = result["video"]["url"]
        video_data = requests.get(video_url).content
        out_path = OUT / "heygen_vertical_test.mp4"
        out_path.write_bytes(video_data)
        print(f"Saved: {out_path} ({len(video_data)/1024:.0f} KB)")
    else:
        print(f"Result keys: {list(result.keys())}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
