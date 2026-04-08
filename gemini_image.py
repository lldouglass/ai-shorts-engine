"""Generate images via Google Gemini API."""
import os
import sys
import base64
import requests
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

def normalize_vertical_image(path: Path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    target_ratio = 9 / 16
    current_ratio = w / h if h else target_ratio

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = max(0, (w - new_w) // 2)
        img = img.crop((left, 0, left + new_w, h))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = max(0, (h - new_h) // 2)
        img = img.crop((0, top, w, top + new_h))

    img = img.resize((1080, 1920), Image.LANCZOS)
    img.save(path, quality=95)


def generate_image(prompt, output_name="gemini_output", model="gemini-2.0-flash-exp"):
    key = os.environ["GOOGLE_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    print(f"Calling Gemini ({model})...")
    resp = requests.post(url, json=payload, timeout=120)
    print(f"Status: {resp.status_code}")

    if resp.status_code != 200:
        print(resp.text[:500])
        return None

    data = resp.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])

    for part in parts:
        if "inlineData" in part:
            img_b64 = part["inlineData"]["data"]
            mime = part["inlineData"]["mimeType"]
            ext = "png" if "png" in mime else "jpg"
            img_bytes = base64.b64decode(img_b64)
            out = Path(f"output/car_videos/{output_name}.{ext}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(img_bytes)
            normalize_vertical_image(out)
            print(f"Saved: {out} ({len(img_bytes)/1024:.0f} KB)")
            return str(out)
        elif "text" in part:
            txt = part["text"]
            print(f"Text response: {txt[:300]}")

    print("No image returned")
    return None


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A cute cartoon cat"
    name = sys.argv[2] if len(sys.argv) > 2 else "gemini_output"
    model = sys.argv[3] if len(sys.argv) > 3 else "nano-banana-pro-preview"
    generate_image(prompt, name, model=model)
