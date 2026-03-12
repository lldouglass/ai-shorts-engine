"""
Car Lifespan Check — Shortform Video Generator (DEFAULT PIPELINE)

End-to-end: car name -> Nano Banana character image -> Edge TTS voiceover ->
HeyGen lip-sync -> post-process (crop 9:16, upscale 1080x1920, captions) -> QA

Usage:
    python make_car_short.py "Range Rover" "I cost a hundred thousand dollars..." "range_rover_unreliable"
    python make_car_short.py --car "Toyota Camry" --script "..." --name "camry_reliable"

Pipeline:
    1. Gemini Nano Banana generates Pixar Cars-style character image
    2. Edge TTS (en-US-AndrewNeural, -5% rate) generates voiceover audio
    3. Preview checkpoint (idea + script + reference image)
    4. fal.ai HeyGen lip-sync (avatar4/image-to-video) creates talking video
    5. post_process.py crops to 9:16, upscales to 1080x1920, overlays captions
    6. video_qa.py validates: resolution, fps, bitrate, lip-sync, captions
    7. Outputs final .mp4 ready for Distribution
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fal_client
import requests
import edge_tts

from gemini_image import generate_image
from post_process import post_process
from video_qa import run_qa

OUT = Path("output/car_videos")
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW_OUT = Path("output/previews")
PREVIEW_OUT.mkdir(parents=True, exist_ok=True)

# Defaults
IMAGE_MODEL = "nano-banana-pro-preview"
TTS_VOICE = "en-US-AndrewNeural"
TTS_RATE = "-5%"
HEYGEN_MODEL = "fal-ai/heygen/avatar4/image-to-video"


def generate_car_image(car_name: str, color: str = "dark green", style_notes: str = "") -> str:
    """Generate a Pixar Cars-style character image via Gemini Nano Banana."""
    safe_name = car_name.lower().replace(" ", "_")
    prompt = (
        f"A Pixar Cars style 3D animated {car_name} character, front view, facing camera directly. "
        f"{color.capitalize()} metallic paint, expressive cartoon eyes on the windshield with eyelids "
        f"and eyebrows, wide talking mouth on the lower bumper/grille showing personality. "
        f"The car looks self-aware and slightly embarrassed, like it knows its own flaws. "
        f"Warm cozy mechanic garage background with tools, spare tires, and warm golden lighting. "
        f"Portrait composition, centered, vertical 9:16 aspect ratio. "
        f"Highly detailed 3D render, Pixar-quality, cinematic lighting. "
        f"The mouth on the bumper MUST be prominent with clear lip definition. "
        f"{style_notes}"
    )
    print(f"\n--- Step 1: Generate {car_name} character image ---")
    img_path = generate_image(prompt, f"{safe_name}_character", model=IMAGE_MODEL)
    if not img_path:
        raise RuntimeError(f"Image generation failed for {car_name}")
    return img_path


def generate_audio(script: str, output_name: str) -> str:
    """Generate TTS voiceover with Edge TTS."""
    print(f"\n--- Step 2: Generate voiceover ({TTS_VOICE}, rate={TTS_RATE}) ---")
    audio_path = str(OUT / f"{output_name}_audio.mp3")

    async def _tts():
        comm = edge_tts.Communicate(script, TTS_VOICE, rate=TTS_RATE)
        await comm.save(audio_path)

    asyncio.run(_tts())
    print(f"Audio saved: {audio_path}")
    return audio_path


def save_preview(idea: str, script: str, image_path: str, output_name: str) -> str:
    """Save a preview card so we always have pre-FAL context logged."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PREVIEW_OUT / f"{output_name}_{ts}_preview.md"
    text = (
        f"# Video Preview\n\n"
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- Idea: {idea}\n"
        f"- Output name: {output_name}\n"
        f"- Image model: {IMAGE_MODEL}\n"
        f"- Reference image: {image_path}\n\n"
        f"## Script\n\n{script}\n"
    )
    path.write_text(text, encoding="utf-8")
    return str(path)


def show_preview(idea: str, script: str, image_path: str, preview_path: str) -> None:
    """Print preview checkpoint and try opening the reference image locally."""
    print("\n--- Step 3: Preview checkpoint (before FAL spend) ---")
    print(f"Idea: {idea}")
    print(f"Script:\n{script}\n")
    print(f"Nano Banana reference image: {image_path}")
    print(f"Preview saved: {preview_path}")

    if os.name == "nt":
        try:
            os.startfile(image_path)  # type: ignore[attr-defined]
            print("Opened reference image preview window.")
        except Exception:
            pass


def should_proceed(auto_approve: bool = False) -> bool:
    """Ask for explicit confirmation before spending FAL credits."""
    if auto_approve:
        print("Auto-approve enabled (--yes). Proceeding to FAL step.")
        return True

    if not sys.stdin.isatty():
        print("Non-interactive terminal and --yes not set. Aborting before FAL spend.")
        return False

    answer = input("Proceed to HeyGen lip-sync and spend FAL credits? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def lipsync_video(image_path: str, audio_path: str, output_name: str) -> str:
    """Upload to fal.ai and run HeyGen lip-sync."""
    print(f"\n--- Step 4: HeyGen lip-sync ---")
    print("Uploading image...")
    image_url = fal_client.upload_file(image_path)
    print(f"Image URL: {image_url}")

    print("Uploading audio...")
    audio_url = fal_client.upload_file(audio_path)
    print(f"Audio URL: {audio_url}")

    print("Running HeyGen lip-sync...")
    result = fal_client.subscribe(
        HEYGEN_MODEL,
        arguments={"image_url": image_url, "audio_url": audio_url},
        with_logs=True,
    )

    video_url = result["video"]["url"]
    raw_path = str(OUT / f"{output_name}_raw.mp4")
    video_data = requests.get(video_url).content
    Path(raw_path).write_bytes(video_data)
    print(f"Raw video: {raw_path} ({len(video_data) / 1024:.0f} KB)")
    return raw_path


def finalize(raw_path: str, script: str, output_name: str) -> dict:
    """Post-process + QA."""
    print(f"\n--- Step 5: Post-process (crop + upscale + captions) ---")
    final_path = str(OUT / f"{output_name}_final.mp4")
    post_process(raw_path, script, output_path=final_path)

    print(f"\n--- Step 6: QA ---")
    qa_result = run_qa(final_path)
    passed = qa_result["passed"]

    if not passed:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\n[FAIL] QA FAILED: {', '.join(failed)}")
    else:
        print(f"\n[PASS] ALL CHECKS PASSED - ready for Distribution")

    return {
        "final_path": final_path,
        "qa_passed": passed,
        "qa_result": qa_result,
    }


def make_car_short(
    car_name: str,
    script: str,
    output_name: str,
    color: str = "dark green",
    style_notes: str = "",
    image_path: str | None = None,
    auto_approve: bool = False,
    preview_only: bool = False,
) -> dict:
    """Full pipeline with pre-FAL preview checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"Making car short: {car_name}")
    print(f"Output: {output_name}")
    print(f"Script: {script[:80]}...")
    print(f"{'=' * 60}")

    if image_path and Path(image_path).exists():
        print(f"\nUsing existing image: {image_path}")
    else:
        image_path = generate_car_image(car_name, color, style_notes)

    audio_path = generate_audio(script, output_name)

    preview_path = save_preview(car_name, script, image_path, output_name)
    show_preview(car_name, script, image_path, preview_path)

    if preview_only:
        return {
            "preview_only": True,
            "idea": car_name,
            "script": script,
            "image_path": image_path,
            "audio_path": audio_path,
            "preview_path": preview_path,
        }

    if not should_proceed(auto_approve=auto_approve):
        return {
            "aborted": True,
            "idea": car_name,
            "script": script,
            "image_path": image_path,
            "audio_path": audio_path,
            "preview_path": preview_path,
            "reason": "User did not confirm FAL spend",
        }

    raw_path = lipsync_video(image_path, audio_path, output_name)
    result = finalize(raw_path, script, output_name)
    result["image_path"] = image_path
    result["audio_path"] = audio_path
    result["raw_path"] = raw_path
    result["preview_path"] = preview_path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Lifespan Check shortform video generator")
    parser.add_argument("car_name", nargs="?", default="Range Rover", help="Car make/model")
    parser.add_argument("script", nargs="?", default=None, help="Voiceover script text")
    parser.add_argument("output_name", nargs="?", default=None, help="Output filename prefix")
    parser.add_argument("--color", default="dark green", help="Car paint color")
    parser.add_argument("--style", default="", help="Extra style notes for image prompt")
    parser.add_argument("--image", default=None, help="Path to existing character image (skip generation)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation and spend FAL credits")
    parser.add_argument("--preview-only", action="store_true", help="Generate image/audio + preview, then stop")
    args = parser.parse_args()

    car = args.car_name
    safe = car.lower().replace(" ", "_")
    script = args.script or f"I'm a {car}. Check your car's lifespan at carlifespancheck.com."
    name = args.output_name or f"{safe}_short"

    result = make_car_short(
        car,
        script,
        name,
        color=args.color,
        style_notes=args.style,
        image_path=args.image,
        auto_approve=args.yes,
        preview_only=args.preview_only,
    )

    if result.get("aborted"):
        print("\nStopped before FAL spend.")
        print(f"Preview: {result['preview_path']}")
        sys.exit(0)

    if result.get("preview_only"):
        print("\nPreview-only complete.")
        print(f"Preview: {result['preview_path']}")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"FINAL: {result['final_path']}")
    print(f"QA: {'PASS' if result['qa_passed'] else 'FAIL'}")
    print(f"Preview: {result['preview_path']}")
    print(f"{'=' * 60}")
