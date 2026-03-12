"""
Dental Education — Shortform Video Generator

Same pipeline as make_car_short.py but for dental content.
Generates a Pixar-style talking tooth character.

Usage:
    python make_tooth_short.py "Flossing Isn't Optional" "script text..." "flossing_mandatory"
    python make_tooth_short.py --topic "Flossing" --script "..." --name "flossing_vid"

Pipeline:
    1. Gemini Nano Banana generates Pixar-style tooth character image
    2. Edge TTS (en-US-AndrewNeural, -5% rate) generates voiceover audio
    3. Preview checkpoint (idea + script + reference image)
    4. fal.ai HeyGen lip-sync creates talking video
    5. post_process.py crops to 9:16, upscales to 1080x1920, overlays captions
    6. video_qa.py validates quality
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

OUT = Path("output/tooth_videos")
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW_OUT = Path("output/previews")
PREVIEW_OUT.mkdir(parents=True, exist_ok=True)

# Defaults (same as car pipeline)
IMAGE_MODEL = "nano-banana-pro-preview"
TTS_VOICE = "en-US-AndrewNeural"
TTS_RATE = "-5%"
HEYGEN_MODEL = "fal-ai/heygen/avatar4/image-to-video"


def generate_tooth_image(topic: str, style_notes: str = "") -> str:
    """Generate a Pixar-style talking tooth character image via Gemini Nano Banana."""
    safe_name = topic.lower().replace(" ", "_")
    prompt = (
        "A Pixar style 3D animated cartoon tooth character, front view, facing camera directly. "
        "Bright white enamel with a subtle pearly sheen, expressive cartoon eyes with eyelids "
        "and eyebrows showing a friendly but serious expression, wide talking mouth showing personality. "
        "The tooth looks like a wise teacher who really cares about dental health. "
        "Clean modern dental office background with soft blue and white tones, dental tools subtly visible, "
        "warm professional lighting. "
        "Portrait composition, centered, vertical 9:16 aspect ratio. "
        "Highly detailed 3D render, Pixar-quality, cinematic lighting. "
        "The mouth MUST be prominent with clear lip definition for lip-sync animation. "
        f"{style_notes}"
    )
    print("\n--- Step 1: Generate tooth character image ---")
    img_path = generate_image(prompt, f"tooth_{safe_name}_character", model=IMAGE_MODEL)
    if not img_path:
        raise RuntimeError("Tooth image generation failed")
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
    print("\n--- Step 4: HeyGen lip-sync ---")
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
    print("\n--- Step 5: Post-process (crop + upscale + captions) ---")
    final_path = str(OUT / f"{output_name}_final.mp4")
    post_process(raw_path, script, output_path=final_path)

    print("\n--- Step 6: QA ---")
    qa_result = run_qa(final_path)
    passed = qa_result["passed"]

    if not passed:
        failed = [r["check"] for r in qa_result["results"] if not r["passed"]]
        print(f"\n[FAIL] QA FAILED: {', '.join(failed)}")
    else:
        print("\n[PASS] ALL CHECKS PASSED - ready for Distribution")

    return {
        "final_path": final_path,
        "qa_passed": passed,
        "qa_result": qa_result,
    }


def make_tooth_short(
    topic: str,
    script: str,
    output_name: str,
    style_notes: str = "",
    image_path: str | None = None,
    auto_approve: bool = False,
    preview_only: bool = False,
) -> dict:
    """Full pipeline with pre-FAL preview checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"Making dental short: {topic}")
    print(f"Output: {output_name}")
    print(f"Script: {script[:80]}...")
    print(f"{'=' * 60}")

    if image_path and Path(image_path).exists():
        print(f"\nUsing existing image: {image_path}")
    else:
        image_path = generate_tooth_image(topic, style_notes)

    audio_path = generate_audio(script, output_name)

    preview_path = save_preview(topic, script, image_path, output_name)
    show_preview(topic, script, image_path, preview_path)

    if preview_only:
        return {
            "preview_only": True,
            "idea": topic,
            "script": script,
            "image_path": image_path,
            "audio_path": audio_path,
            "preview_path": preview_path,
        }

    if not should_proceed(auto_approve=auto_approve):
        return {
            "aborted": True,
            "idea": topic,
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


# Pre-built dental scripts
DENTAL_SCRIPTS = {
    "flossing": (
        "Flossing Isn't Optional",
        "Hey, I need to tell you something your dentist wishes you'd listen to. "
        "When you only brush, you're cleaning about 65 percent of your tooth surface. "
        "That means 35 percent of every single tooth in your mouth never gets touched. "
        "That's where cavities start. That's where gum disease starts. "
        "Flossing takes two minutes. Two minutes to reach the spots your toothbrush literally cannot get to. "
        "Your gums might bleed the first week. That's normal. It means they need it. "
        "After a week of daily flossing, the bleeding stops because your gums get healthier. "
        "So do yourself a favor. Floss tonight. Your teeth will thank you."
    ),
    "rinsing": (
        "Stop Rinsing After You Brush",
        "OK here's something most people get wrong. You brush your teeth, then you rinse your mouth with water. "
        "Sounds normal right? It's actually the worst thing you can do. "
        "Your toothpaste has fluoride in it. Fluoride protects your enamel and fights cavities. "
        "When you rinse right after brushing, you wash all that fluoride away before it can work. "
        "Instead, just spit out the extra toothpaste. Don't rinse. Let the fluoride sit on your teeth. "
        "It needs time to absorb and do its job. "
        "This one simple change can seriously reduce your cavity risk. Just spit, don't rinse."
    ),
    "brushing_hard": (
        "You're Brushing Too Hard",
        "I see it all the time. People think brushing harder means cleaner teeth. "
        "It doesn't. It means damaged teeth. "
        "When you press too hard, you wear down your enamel. That's the protective layer on your teeth. "
        "Once enamel is gone, it doesn't grow back. Ever. "
        "You also push your gums back, exposing the sensitive root underneath. "
        "That's why your teeth feel sensitive to cold drinks and ice cream. "
        "Use a soft bristle brush. Hold it like a pencil, not a hammer. "
        "Gentle circles, two minutes, twice a day. That's all you need."
    ),
    "ice_chewing": (
        "Ice Chewing Is Worse Than You Think",
        "If you chew ice, I need you to stop. Like, today. "
        "Every time you crunch down on ice, you're creating tiny micro cracks in your enamel. "
        "You can't see them. You can't feel them. But they're there. "
        "And one day, you bite down on something normal and your tooth just breaks. "
        "I see it happen all the time. A perfectly healthy tooth, split in half, because of years of ice chewing. "
        "That's a crown. Maybe a root canal. Hundreds or thousands of dollars. "
        "All because of a habit you thought was harmless. "
        "If you like cold stuff, drink cold water. Just don't chew the ice."
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dental education shortform video generator")
    parser.add_argument(
        "topic",
        nargs="?",
        default="flossing",
        help="Topic name or key from DENTAL_SCRIPTS (flossing, rinsing, brushing_hard, ice_chewing)",
    )
    parser.add_argument("script", nargs="?", default=None, help="Custom voiceover script text")
    parser.add_argument("output_name", nargs="?", default=None, help="Output filename prefix")
    parser.add_argument("--style", default="", help="Extra style notes for image prompt")
    parser.add_argument("--image", default=None, help="Path to existing character image (skip generation)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation and spend FAL credits")
    parser.add_argument("--preview-only", action="store_true", help="Generate image/audio + preview, then stop")
    args = parser.parse_args()

    topic_key = args.topic.lower().replace(" ", "_")

    if topic_key in DENTAL_SCRIPTS and not args.script:
        display_topic, script = DENTAL_SCRIPTS[topic_key]
    else:
        display_topic = args.topic
        script = args.script or "Take care of your teeth. Visit your dentist regularly."

    name = args.output_name or f"tooth_{topic_key}_short"

    result = make_tooth_short(
        display_topic,
        script,
        name,
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
