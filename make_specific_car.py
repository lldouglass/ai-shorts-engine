"""Generate a specific car model character + video."""
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import fal_client
import requests
import edge_tts
from pathlib import Path
from post_process import post_process

OUT = Path("output/car_videos")
OUT.mkdir(parents=True, exist_ok=True)


def generate_car_character(car_name, color, style_notes=""):
    """Generate a specific car character image."""
    client = OpenAI()
    prompt = (
        f"A 3D animated cartoon {car_name} character in Pixar-quality animation style, NOT any existing character. "
        f"An original friendly {car_name} with a face on the front. "
        "The windshield is the eyes - large, expressive, emotive eyes with eyelids and eyebrows. "
        "The front bumper IS the mouth - a wide, clearly defined mouth with visible lips, slightly open mid-speech. "
        "The mouth must be large, prominent, capable of showing expressions. "
        f"{color} paint with chrome accents. {style_notes} "
        "Setting: warm cozy garage with tools and car parts. Warm golden lighting. "
        "The car has personality - self-aware, slightly embarrassed, like it knows its own flaws. "
        "Portrait composition, centered, 9:16 vertical aspect ratio. "
        "Highly detailed 3D render, Pixar-quality, cinematic lighting. "
        "The mouth on the bumper MUST be very prominent with clear lip definition for lip-sync."
    )

    print(f"Generating {car_name} character...")
    response = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1792", quality="hd", n=1)
    img_data = requests.get(response.data[0].url).content
    safe_name = car_name.lower().replace(" ", "_")
    img_path = OUT / f"{safe_name}_character.png"
    img_path.write_bytes(img_data)
    print(f"Saved: {img_path} ({len(img_data)/1024:.0f} KB)")
    return str(img_path)


def make_specific_car_video(car_name, script_text, output_name, color="dark green", style_notes="",
                            title=None, final_caption=None):
    """Full pipeline for a specific car model video."""
    print(f"\n{'='*50}")
    print(f"Making {car_name} video: {output_name}")
    print(f"Script: {script_text}")
    print(f"{'='*50}\n")

    # 1. Generate character
    image_path = generate_car_character(car_name, color, style_notes)

    # 2. Audio (British mechanic voice)
    print("Generating audio...")
    audio_path = str(OUT / f"{output_name}_audio.mp3")
    async def _tts():
        comm = edge_tts.Communicate(script_text, "en-GB-RyanNeural")
        await comm.save(audio_path)
    asyncio.run(_tts())

    # 3. Lip-sync
    print("Uploading to fal.ai...")
    image_url = fal_client.upload_file(image_path)
    audio_url = fal_client.upload_file(audio_path)
    print("Running HeyGen lip-sync...")
    result = fal_client.subscribe(
        "fal-ai/heygen/avatar4/image-to-video",
        arguments={"image_url": image_url, "audio_url": audio_url},
        with_logs=True,
    )
    raw_path = str(OUT / f"{output_name}_raw.mp4")
    video_data = requests.get(result["video"]["url"]).content
    Path(raw_path).write_bytes(video_data)
    print(f"Raw video: {raw_path} ({len(video_data)/1024:.0f} KB)")

    # 4. Post-process
    final_path = str(OUT / f"{output_name}_final.mp4")
    post_process(raw_path, script_text, output_path=final_path)

    # 5. QA
    print("\nRunning QA...")
    from video_qa import run_qa
    qa_result = run_qa(final_path)
    if not qa_result["passed"]:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\nQA FAILED: {', '.join(failed)}")
        return {"final_video_path": final_path, "qa_passed": False, "qa_result": qa_result}

    # 6. Distribution header
    from make_candle_video import get_video_metadata, build_distribution_caption
    if title is None:
        title = output_name.replace("_", " ").title()
    if final_caption is None:
        final_caption = script_text
    meta = get_video_metadata(final_path)
    header = build_distribution_caption(
        brand="carlifespancheck", title=title,
        source_file=meta["source_file"], width=meta["width"], height=meta["height"],
        fps=meta["fps"], bitrate_mbps=meta["bitrate_mbps"], final_caption=final_caption,
    )
    print("\n--- READY_FOR_DISTRIBUTION ---")
    print(header)

    return {"final_video_path": final_path, "distribution_header": header, "metadata": meta, "qa_passed": True}


if __name__ == "__main__":
    import sys
    make_specific_car_video(
        car_name=sys.argv[1] if len(sys.argv) > 1 else "Range Rover",
        script_text=sys.argv[2] if len(sys.argv) > 2 else "Test script",
        output_name=sys.argv[3] if len(sys.argv) > 3 else "test_car",
        color=sys.argv[4] if len(sys.argv) > 4 else "dark green",
    )
