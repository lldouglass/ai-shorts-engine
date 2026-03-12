"""Test Veo 3.1 image-to-video with the Range Rover character."""
import os
import base64
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ["GOOGLE_API_KEY"]
OUT = Path("output/car_videos")


def veo_image_to_video(image_path, prompt, model="veo-3.1-generate-preview",
                       aspect_ratio="9:16", duration=8):
    """Generate video from image using Veo via Gemini API."""
    
    # Read and encode image
    img_bytes = Path(image_path).read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Determine mime type
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predictLongRunning"
    headers = {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    
    payload = {
        "instances": [
            {
                "prompt": prompt,
                "image": {
                    "mimeType": mime,
                    "bytesBase64Encoded": img_b64,
                },
            }
        ],
        "parameters": {
            "aspectRatio": aspect_ratio,
            "resolution": "720p",
            "durationSeconds": duration,
            "sampleCount": 1,
            # "includeAudio": True,  # Not supported on preview models
        },
    }
    
    print(f"Submitting Veo job ({model})...")
    print(f"  Image: {image_path} ({len(img_bytes)/1024:.0f} KB)")
    print(f"  Prompt: {prompt[:100]}...")
    print(f"  Aspect: {aspect_ratio}, Duration: {duration}s")
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"  Error: {resp.text[:500]}")
        return None
    
    result = resp.json()
    operation_name = result.get("name")
    print(f"  Operation: {operation_name}")
    
    # Poll for completion
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}"
    print("  Polling for completion...")
    
    for i in range(60):  # Max 5 minutes
        time.sleep(5)
        poll_resp = requests.get(poll_url, headers={"x-goog-api-key": API_KEY}, timeout=30)
        poll_data = poll_resp.json()
        
        done = poll_data.get("done", False)
        if done:
            print(f"  Done! (after {(i+1)*5}s)")
            
            # Extract video
            response = poll_data.get("response", {})
            videos = response.get("videos", response.get("generateVideoResponse", {}).get("generatedSamples", []))
            
            if not videos:
                # Try alternate response structure
                print(f"  Response keys: {list(poll_data.keys())}")
                print(f"  Response: {str(poll_data)[:500]}")
                return None
            
            video_data = videos[0] if isinstance(videos, list) else videos
            
            # Check for video bytes or URL
            if "video" in video_data:
                vid = video_data["video"]
                if "bytesBase64Encoded" in vid:
                    video_bytes = base64.b64decode(vid["bytesBase64Encoded"])
                    out_path = OUT / "veo_range_rover_raw.mp4"
                    out_path.write_bytes(video_bytes)
                    print(f"  Saved: {out_path} ({len(video_bytes)/1024:.0f} KB)")
                    return str(out_path)
                elif "uri" in vid:
                    video_url = vid["uri"]
                    video_bytes = requests.get(video_url).content
                    out_path = OUT / "veo_range_rover_raw.mp4"
                    out_path.write_bytes(video_bytes)
                    print(f"  Saved: {out_path} ({len(video_bytes)/1024:.0f} KB)")
                    return str(out_path)
            
            # Try flat structure
            if "bytesBase64Encoded" in video_data:
                video_bytes = base64.b64decode(video_data["bytesBase64Encoded"])
                out_path = OUT / "veo_range_rover_raw.mp4"
                out_path.write_bytes(video_bytes)
                print(f"  Saved: {out_path} ({len(video_bytes)/1024:.0f} KB)")
                return str(out_path)
            
            print(f"  Video data keys: {list(video_data.keys()) if isinstance(video_data, dict) else type(video_data)}")
            print(f"  Full response: {str(poll_data)[:1000]}")
            return None
        
        # Show progress
        metadata = poll_data.get("metadata", {})
        progress = metadata.get("progressPercent", "?")
        if i % 6 == 0:
            print(f"  ...waiting ({(i+1)*5}s, progress: {progress}%)")
    
    print("  Timed out after 5 minutes")
    return None


if __name__ == "__main__":
    video_path = veo_image_to_video(
        image_path="output/car_videos/range_rover_gemini_ui.png",
        prompt=(
            "The animated Pixar-style Range Rover character talks directly to the camera. "
            "Its mouth on the bumper opens and closes as it speaks, with natural lip movements. "
            "Its windshield eyes blink and express embarrassment. "
            "The car says: 'I cost a hundred thousand dollars and I scored a two out of five on reliability. "
            "My air suspension fails, my electronics glitch, and I have been recalled five times. "
            "But hey, at least I look good in the shop.' "
            "The car speaks with a British accent. Warm garage lighting. Subtle camera movement."
        ),
        aspect_ratio="9:16",
        duration=8,
    )
    
    if video_path:
        print(f"\nSuccess! Video at: {video_path}")
    else:
        print("\nFailed to generate video")
