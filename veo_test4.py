"""Veo test: strict identity lock while animating input image."""
import os, base64, time, requests
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ["GOOGLE_API_KEY"]
img_path = Path("output/car_videos/range_rover_gemini_ui.png")
img_b64 = base64.b64encode(img_path.read_bytes()).decode()

url = "https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-generate-preview:predictLongRunning"
headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}

payload = {
    "instances": [{
        "prompt": (
            "Animate THIS EXACT input image. Keep the same car character identity, same model, same color, "
            "same garage background, same camera angle and composition. Do not replace or morph the subject. "
            "Only add subtle animation: mouth opens/closes as if speaking, eyes blink, tiny head/body bounce. "
            "Maintain exact visual consistency with the provided image."
        ),
        "image": {"mimeType": "image/png", "bytesBase64Encoded": img_b64},
    }],
    "parameters": {
        "aspectRatio": "9:16",
        "resolution": "720p",
        "durationSeconds": 8,
        "sampleCount": 1,
    },
}

print("Submitting Veo job (identity lock)...")
r = requests.post(url, headers=headers, json=payload, timeout=60)
print("Status:", r.status_code)
if r.status_code != 200:
    print(r.text[:800])
    raise SystemExit(1)

op = r.json()["name"]
print("Operation:", op)

for i in range(70):
    time.sleep(5)
    s = requests.get(f"https://generativelanguage.googleapis.com/v1beta/{op}", headers={"x-goog-api-key": API_KEY}, timeout=30).json()
    if s.get("done"):
        print(f"Done after {(i+1)*5}s")
        gen = s.get("response", {}).get("generateVideoResponse", {})
        if gen.get("raiMediaFilteredCount", 0):
            print("BLOCKED:", gen.get("raiMediaFilteredReasons", []))
            raise SystemExit(1)
        samples = gen.get("generatedSamples", [])
        if not samples:
            print("No samples in response")
            print(str(s)[:1500])
            raise SystemExit(1)
        vid = samples[0].get("video", {})
        out = Path("output/car_videos/veo_rr_identity_raw.mp4")
        if "bytesBase64Encoded" in vid:
            out.write_bytes(base64.b64decode(vid["bytesBase64Encoded"]))
        elif "uri" in vid:
            data = requests.get(vid["uri"], headers={"x-goog-api-key": API_KEY}, timeout=60).content
            if len(data) < 1000:
                sep = "&" if "?" in vid["uri"] else "?"
                data = requests.get(vid["uri"] + f"{sep}key={API_KEY}", timeout=60).content
            out.write_bytes(data)
        else:
            print("Unknown video format:", vid.keys())
            raise SystemExit(1)
        print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")
        break
    if i % 6 == 0:
        print(f"...waiting ({(i+1)*5}s)")
else:
    print("Timed out")
    raise SystemExit(1)
