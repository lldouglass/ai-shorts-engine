"""Generate candle character image + hook audio for lip-sync test."""
import os
import asyncio
from openai import OpenAI
from pathlib import Path
import requests

OUT = Path("output/hook_test")
OUT.mkdir(parents=True, exist_ok=True)

# --- 1. Generate candle character image with DALL-E ---
def generate_candle_image():
    client = OpenAI()
    
    prompt = (
        "A photorealistic 3D rendered anthropomorphic candle character sitting in a cozy library. "
        "The candle has a warm, friendly face with clear expressive brown eyes, a small nose, "
        "and a visible mouth suitable for talking/lip-sync animation. The candle is cream/ivory colored "
        "with a gentle flame on top. Warm bokeh lighting, wooden bookshelves in background. "
        "The candle has a wise, slightly smug expression like a knowledgeable professor. "
        "Pixar/Disney animation style. Portrait composition, centered, 9:16 vertical aspect ratio. "
        "High detail, cinematic lighting."
    )
    
    print("Generating candle character image...")
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1792",
        quality="hd",
        n=1,
    )
    
    image_url = response.data[0].url
    revised_prompt = response.data[0].revised_prompt
    print(f"Revised prompt: {revised_prompt}")
    
    # Download
    img_data = requests.get(image_url).content
    img_path = OUT / "candle_character.png"
    img_path.write_bytes(img_data)
    print(f"Saved: {img_path} ({len(img_data)/1024:.0f} KB)")
    return img_path


# --- 2. Generate hook audio with Edge TTS ---
async def generate_hook_audio():
    import edge_tts
    
    # Punchy Munger-style hook
    hook_text = "Most investors lose money because they're trying to be smart. The real edge? Just avoid being stupid."
    
    voice = "en-US-AndrewNeural"  # Preferred voice from TOOLS.md
    
    print(f"Generating hook audio with {voice}...")
    communicate = edge_tts.Communicate(hook_text, voice, rate="-5%")
    audio_path = str(OUT / "hook_audio.mp3")
    await communicate.save(audio_path)
    print(f"Saved: {audio_path}")
    return audio_path


if __name__ == "__main__":
    # Generate image
    img_path = generate_candle_image()
    
    # Generate audio
    audio_path = asyncio.run(generate_hook_audio())
    
    print(f"\n--- Ready for lip-sync ---")
    print(f"Image: {img_path}")
    print(f"Audio: {audio_path}")
    print(f"\nNext: upload both to Hedra.com for lip-sync test")
