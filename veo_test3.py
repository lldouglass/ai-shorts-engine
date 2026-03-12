"""Test Veo with simple animate prompt."""
import os, base64, time, requests
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ["GOOGLE_API_KEY"]
img_bytes = Path("output/car_videos/range_rover_gemini_ui.png").read_bytes()
img_b64 = base64.b64encode(img_bytes).decode()

url = "https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-generate-preview:predictLongRunning"
headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}

payload = {
    "instances": [{
        "prompt": "Animate this picture. The car character comes to life and talks to the camera.",
        "image": {"mimeType": "image/png", "bytesBase64Encoded": img_b64},
    }],
    "parameters": {
        "aspectRatio": "9:16",
        "resolution": "720p",
        "durationSeconds": 8,
        "sampleCount": 1,
    },
}

print("Submitting Veo job (simple animate prompt)...")
resp = requests.post(url, headers=headers, json=payload, timeout=60)
print(f"Status: {resp.status_code}")
if resp.status_code != 200:
    print(resp.text[:500])
    exit(1)

op = resp.json()["name"]
print(f"Operation: {op}")

for i in range(60):
    time.sleep(5)
    r = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/{op}",
        headers={"x-goog-api-key": API_KEY},
    ).json()
    
    if r.get("done"):
        print(f"Done after {(i+1)*5}s")
        gen = r.get("response", {}).get("generateVideoResponse", {})
        
        filtered = gen.get("raiMediaFilteredCount", 0)
        if filtered:
            reasons = gen.get("raiMediaFilteredReasons", [])
            print(f"BLOCKED: {reasons}")
            break
        
        samples = gen.get("generatedSamples", [])
        if samples:
            vid = samples[0].get("video", {})
            if "uri" in vid:
                vb = requests.get(vid["uri"], headers={"x-goog-api-key": API_KEY}).content
                if len(vb) < 1000:
                    vb = requests.get(vid["uri"] + ("&" if "?" in vid["uri"] else "?") + f"key={API_KEY}").content
                out = Path("output/car_videos/veo_rr_simple_raw2.mp4")
                out.write_bytes(vb)
                print(f"Saved: {out} ({len(vb)/1024:.0f} KB)")
            elif "bytesBase64Encoded" in vid:
                vb = base64.b64decode(vid["bytesBase64Encoded"])
                out = Path("output/car_videos/veo_rr_simple_raw2.mp4")
                out.write_bytes(vb)
                print(f"Saved: {out} ({len(vb)/1024:.0f} KB)")
        else:
            print(f"No samples")
            import json
            print(json.dumps(r, indent=2, default=str)[:800])
        break
    
    if i % 6 == 0:
        print(f"...waiting ({(i+1)*5}s)")

else:
    print("Timed out")
