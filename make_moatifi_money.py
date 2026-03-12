"""Moatifi video: talking money character + Munger quote.

Uses same proven pipeline as car shorts:
Nano Banana image → Edge TTS → HeyGen lip-sync → post-process → QA
"""
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

import fal_client
import requests
import edge_tts
from pathlib import Path
from gemini_image import generate_image
from post_process import post_process
from video_qa import run_qa

OUT = Path("output/moatifi_videos")
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_MODEL = "nano-banana-pro-preview"
TTS_VOICE = "en-US-AndrewNeural"
TTS_RATE = "-5%"
OUTPUT_NAME = "munger_envy"

# Munger quote adapted as first-person talking money monologue
SCRIPT = (
    "Charlie Munger said the world is not driven by greed. It is driven by envy. "
    "And he was right. Most investors do not lose money because they are dumb. "
    "They lose money because they saw someone else get rich and could not stand being left out. "
    "That is not a strategy. That is a feeling. And feelings make terrible portfolio managers."
)


def main():
    # 1. Generate talking money character
    print("--- Step 1: Generate talking money character ---")
    prompt = (
        "A Pixar-style 3D animated one hundred dollar bill character, front view, facing camera. "
        "The bill is standing upright with small cartoon arms and legs. "
        "Benjamin Franklin's face on the bill is alive with large expressive cartoon eyes, "
        "raised eyebrows, and a wide talking mouth mid-speech. "
        "He looks wise, slightly smug, like he's about to drop some knowledge. "
        "Background: a luxurious dark wood office with bookshelves, leather chair, and warm lamp light. "
        "Portrait composition, centered, vertical 9:16 aspect ratio. "
        "Highly detailed 3D render, Pixar-quality, cinematic warm lighting. "
        "The mouth MUST be prominent with clear lip definition for lip-sync animation."
    )
    img_path = generate_image(prompt, f"{OUTPUT_NAME}_character", model=IMAGE_MODEL)
    if not img_path:
        raise RuntimeError("Image generation failed")
    print(f"Image: {img_path}")

    # 2. TTS voiceover
    print(f"\n--- Step 2: Generate voiceover ({TTS_VOICE}, {TTS_RATE}) ---")
    audio_path = str(OUT / f"{OUTPUT_NAME}_audio.mp3")

    async def _tts():
        comm = edge_tts.Communicate(SCRIPT, TTS_VOICE, rate=TTS_RATE)
        await comm.save(audio_path)
    asyncio.run(_tts())
    print(f"Audio: {audio_path}")

    # 3. HeyGen lip-sync
    print("\n--- Step 3: HeyGen lip-sync ---")
    print("Uploading image...")
    image_url = fal_client.upload_file(img_path)
    print("Uploading audio...")
    audio_url = fal_client.upload_file(audio_path)
    print("Running HeyGen...")
    result = fal_client.subscribe(
        "fal-ai/heygen/avatar4/image-to-video",
        arguments={"image_url": image_url, "audio_url": audio_url},
        with_logs=True,
    )
    video_url = result["video"]["url"]
    raw_path = str(OUT / f"{OUTPUT_NAME}_raw.mp4")
    video_data = requests.get(video_url).content
    Path(raw_path).write_bytes(video_data)
    print(f"Raw video: {raw_path} ({len(video_data)/1024:.0f} KB)")

    # 4. Post-process
    print("\n--- Step 4: Post-process ---")
    final_path = str(OUT / f"{OUTPUT_NAME}_final.mp4")
    post_process(raw_path, SCRIPT, output_path=final_path)

    # 5. QA
    print("\n--- Step 5: QA ---")
    qa_result = run_qa(final_path)
    if qa_result["passed"]:
        print("\n[PASS] Ready for Distribution")
    else:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\n[FAIL] {', '.join(failed)}")

    return final_path


if __name__ == "__main__":
    main()
