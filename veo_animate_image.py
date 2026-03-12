"""Animate an input image with Veo (Gemini API predictLongRunning)."""
import os
import sys
import time
import base64
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]


def animate(image_path: str, out_path: str, prompt: str, model: str = "veo-2.0-generate-001", duration: int = 8):
    img_bytes = Path(image_path).read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predictLongRunning"
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}

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
            "aspectRatio": "9:16",
            "resolution": "720p",
            "durationSeconds": duration,
            "sampleCount": 1,
        },
    }

    print(f"Submitting {model} for {Path(image_path).name}...")
    r = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print(r.text[:1200])
        return False

    op_name = r.json().get("name")
    print("Operation:", op_name)

    op_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}"
    for i in range(90):
        time.sleep(5)
        s = requests.get(op_url, headers={"x-goog-api-key": API_KEY}, timeout=60).json()
        if s.get("done"):
            Path("output/car_videos/veo_last_response.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
            gen = s.get("response", {}).get("generateVideoResponse", {})

            if gen.get("raiMediaFilteredCount", 0):
                print("BLOCKED:", gen.get("raiMediaFilteredReasons", []))
                return False

            samples = gen.get("generatedSamples", [])
            if not samples:
                print("No samples in response")
                print(json.dumps(s, indent=2)[:1500])
                return False

            video = samples[0].get("video", {})
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            if "bytesBase64Encoded" in video:
                out.write_bytes(base64.b64decode(video["bytesBase64Encoded"]))
                print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")
                return True

            if "uri" in video:
                uri = video["uri"]
                data = requests.get(uri, headers={"x-goog-api-key": API_KEY}, timeout=120).content
                # fallback for uri requiring key query param
                if len(data) < 1024:
                    sep = "&" if "?" in uri else "?"
                    data = requests.get(uri + f"{sep}key={API_KEY}", timeout=120).content
                out.write_bytes(data)
                print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")
                return out.stat().st_size > 1024

            print("Unknown video payload keys:", list(video.keys()))
            return False

        if i % 6 == 0:
            print(f"...waiting ({(i+1)*5}s)")

    print("Timed out")
    return False


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "output/car_videos/range_rover_nb_v3.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/car_videos/veo2_rr_from_nb_raw.mp4"
    model = sys.argv[3] if len(sys.argv) > 3 else "veo-2.0-generate-001"
    prompt = (
        "Animate this exact input image. Keep the same character identity, same car shape, same color and same scene. "
        "Only add subtle motion: the mouth opens and closes as if talking, eyes blink, and slight body bounce. "
        "No scene changes and no object replacement."
    )
    ok = animate(image, out, prompt, model=model, duration=8)
    raise SystemExit(0 if ok else 1)
