"""Generate talking car character image + video for Car Lifespan Check."""
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

CAR_IMAGE = OUT / "car_character.png"


def generate_car_character():
    """Generate a talking car character image with DALL-E."""
    if CAR_IMAGE.exists():
        print(f"Car character already exists: {CAR_IMAGE}")
        return str(CAR_IMAGE)

    client = OpenAI()
    prompt = (
        "A photorealistic 3D rendered anthropomorphic vintage car character in a mechanic's garage. "
        "The car is a friendly classic sedan with a warm face on the front - "
        "large expressive headlight-eyes, a wide bumper-mouth that is slightly open as if talking, "
        "with defined lips/grille that contrast clearly against the body. "
        "The car has a wise, slightly cheeky expression like a trusted old mechanic who knows everything. "
        "Cherry red paint with chrome accents. Warm workshop lighting, tool boards and tires in background. "
        "Pixar/Disney Cars movie animation style. Portrait composition, centered, 9:16 vertical aspect ratio. "
        "High detail, cinematic lighting. The mouth/grille must be prominent and well-defined for lip-sync."
    )

    print("Generating car character image...")
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1792",
        quality="hd",
        n=1,
    )

    img_data = requests.get(response.data[0].url).content
    CAR_IMAGE.write_bytes(img_data)
    print(f"Saved: {CAR_IMAGE} ({len(img_data)/1024:.0f} KB)")
    print(f"Revised prompt: {response.data[0].revised_prompt}")
    return str(CAR_IMAGE)


def make_car_video(script_text, output_name, title=None, final_caption=None):
    """Full pipeline for car website video."""
    print(f"\n{'='*50}")
    print(f"Making car video: {output_name}")
    print(f"Script: {script_text}")
    print(f"{'='*50}\n")

    # 1. Ensure car character exists
    image_path = generate_car_character()

    # 2. Generate audio
    print("Generating audio...")
    audio_path = str(OUT / f"{output_name}_audio.mp3")

    async def _tts():
        comm = edge_tts.Communicate(script_text, "en-GB-RyanNeural")
        await comm.save(audio_path)
    asyncio.run(_tts())
    print(f"Audio: {audio_path}")

    # 3. Upload + lip-sync
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

    # 4. Post-process (crop + upscale + captions)
    final_path = str(OUT / f"{output_name}_final.mp4")
    post_process(raw_path, script_text, output_path=final_path)

    # 5. Run QA
    print("\nRunning QA checks...")
    from video_qa import run_qa
    qa_result = run_qa(final_path)
    if not qa_result["passed"]:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\nQA FAILED: {', '.join(failed)}")
        print("Video NOT sent to Distribution. Fix issues and retry.")
        return {"final_video_path": final_path, "qa_passed": False, "qa_result": qa_result}

    # 6. Build distribution header
    from make_candle_video import get_video_metadata, build_distribution_caption
    if title is None:
        title = output_name.replace("_", " ").title()
    if final_caption is None:
        final_caption = script_text

    meta = get_video_metadata(final_path)
    header = build_distribution_caption(
        brand="carlifespancheck",
        title=title,
        source_file=meta["source_file"],
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        bitrate_mbps=meta["bitrate_mbps"],
        final_caption=final_caption,
    )

    header_path = OUT / f"{output_name}_distribution_header.txt"
    header_path.write_text(header, encoding="utf-8")

    print(f"\nFinal: {final_path}")
    print("\n--- READY_FOR_DISTRIBUTION ---")
    print(header)

    return {"final_video_path": final_path, "distribution_header": header, "metadata": meta, "qa_passed": True}


if __name__ == "__main__":
    import sys
    script = sys.argv[1] if len(sys.argv) > 1 else "Your car is trying to tell you something. Most people just aren't listening."
    name = sys.argv[2] if len(sys.argv) > 2 else "car_video"
    make_car_video(script, name)
