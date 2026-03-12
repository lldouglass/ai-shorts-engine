"""
Multi-Character Listicle Video Pipeline

4 animated characters each explain one point on a topic.
5-8 seconds per character. Total: 20-32 seconds.
Default audio stack:
- BGM random rotation from music_v2/
- SFX timing map (intro pop + transition whoosh + key-point impacts)
- Voice-first mix (BGM/SFX kept under narration)

Usage:
    python make_listicle_video.py --brand car
    python make_listicle_video.py --brand car --segments segments.json
    python make_listicle_video.py --brand moatifi --skip-images  (reuse existing images)
"""
import os
import sys
import json
import asyncio
import random
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import fal_client
import requests
import edge_tts
import imageio_ffmpeg
from moviepy import VideoFileClip

from gemini_image import generate_image
from post_process import crop_to_vertical, add_captions

OUT = Path("output/listicle_videos")
OUT.mkdir(parents=True, exist_ok=True)

HEYGEN_MODEL = "fal-ai/heygen/avatar4/image-to-video"
TTS_RATE = "-5%"
IMAGE_MODEL = "nano-banana-pro-preview"

# Audio defaults (approved Mar 9)
BGM_VOLUME = 0.12
SFX_INTRO_VOLUME = 0.22
SFX_TRANSITION_VOLUME = 0.26
SFX_IMPACT_VOLUME = 0.24

# SFX density profile (Logan feedback: previous stack had too many events)
SFX_USE_INTRO = False
SFX_MAX_TRANSITIONS = 1
SFX_MAX_IMPACTS = 1

# Preferred BGM pool (v2)
BGM_POOL = [
    Path("music_v2/bgm_v2_driving_ambition.mp3"),
    Path("music_v2/bgm_v2_dirty_thinkin.mp3"),
    Path("music_v2/bgm_v2_cat_walk.mp3"),
    Path("music_v2/bgm_v2_discover.mp3"),
    Path("music_v2/bgm_v2_sports_highlights.mp3"),
    Path("music_v2/bgm_v2_night_sky_hiphop.mp3"),
    Path("music_v2/bgm_v2_tech_house.mp3"),
    Path("music_v2/bgm_v2_deep_urban.mp3"),
]

# Fallback BGM if v2 pool missing
BGM_FALLBACKS = [
    Path("music/bgm_energetic.mp3"),
    Path("music/bgm_dramatic.mp3"),
    Path("music/bgm_suspense.mp3"),
    Path("music/bgm_chill.mp3"),
    Path("music/bgm_motivational.mp3"),
    Path("ambient_pad.wav"),
]

# SFX defaults
SFX_INTRO_POOL = [
    Path("sfx/pop_1.mp3"),
    Path("sfx/ding_1.mp3"),
]
SFX_TRANSITION_POOL = [
    Path("sfx/whoosh_2.mp3"),
    Path("sfx/swoosh_1.mp3"),
    Path("sfx/transition_1.mp3"),
    Path("sfx/slide_1.mp3"),
]
SFX_IMPACT_POOL = [
    Path("sfx/impact_1.mp3"),
    Path("sfx/impact_2.mp3"),
    Path("sfx/ding_2.mp3"),
]


def get_ffmpeg_binary() -> str:
    """Return a working ffmpeg executable path."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def pick_existing(paths):
    """Return one existing path from a pool, or None."""
    existing = [Path(p) for p in paths if Path(p).exists()]
    return random.choice(existing) if existing else None


# ── Test Content ─────────────────────────────────────────────────
CAR_SEGMENTS = [
    {
        "character": "Range Rover",
        "color": "black",
        "voice": "en-US-AndrewNeural",
        "script": (
            "I'm beautiful, I know. But after a hundred thousand miles? "
            "Air suspension fails, engine overheats, transfer case dies. "
            "Average repair bill? Forty-five hundred a year. Sorry about that."
        ),
        "expression": "apologetic but smug, slight smirk",
    },
    {
        "character": "BMW X5",
        "color": "silver",
        "voice": "en-GB-RyanNeural",
        "script": (
            "My timing chain? Three thousand to fix. "
            "Coolant system? Basically a ticking time bomb. "
            "I'm German engineered, to keep your mechanic employed."
        ),
        "expression": "arrogant and unapologetic, raised eyebrow",
    },
    {
        "character": "Jeep Grand Cherokee",
        "color": "dark red",
        "voice": "en-US-GuyNeural",
        "script": (
            "I look rugged but my electronics quit around eighty thousand miles. "
            "Transmission problems, electrical gremlins, and this screen "
            "freezes more than a Minnesota winter."
        ),
        "expression": "tough but embarrassed, wincing",
    },
    {
        "character": "Nissan Pathfinder",
        "color": "dark gray",
        "voice": "en-US-ChristopherNeural",
        "script": (
            "My CVT transmission has a shorter lifespan than most celebrity marriages. "
            "Four grand to replace and it's not if, it's when. "
            "Which one of these drained your wallet? Drop it below."
        ),
        "expression": "honest and defeated, sad eyes",
    },
]

MOATIFI_SEGMENTS = [
    {
        "character": "a golden trophy that's cracked and tarnished",
        "color": "gold",
        "voice": "en-US-AndrewNeural",
        "script": (
            "I'm a value trap. My PE ratio looks cheap but my earnings "
            "are falling off a cliff. Low price doesn't mean good deal. "
            "It means nobody else wants me either."
        ),
        "expression": "proud but crumbling, fake smile",
    },
    {
        "character": "a piggy bank with a bandage on it",
        "color": "pink",
        "voice": "en-GB-RyanNeural",
        "script": (
            "They bought me because I paid a fat dividend. "
            "Then the company cut it by sixty percent. "
            "Chasing yield without checking the payout ratio? Classic mistake."
        ),
        "expression": "injured and annoyed",
    },
    {
        "character": "a melting ice cube wearing a business suit",
        "color": "light blue",
        "voice": "en-US-GuyNeural",
        "script": (
            "I'm a hype stock. Went up three hundred percent in six months. "
            "No moat, no profits, just vibes. Now I'm down eighty percent "
            "and people call me a long term hold. I'm not. I'm melting."
        ),
        "expression": "panicked and melting, desperate eyes",
    },
    {
        "character": "a rusty old safe that's wide open and empty",
        "color": "rusty brown",
        "voice": "en-US-ChristopherNeural",
        "script": (
            "I looked rock solid. Fortress balance sheet, right? "
            "Turns out my debt was hidden in off-balance-sheet tricks. "
            "What's the worst stock that fooled you? Tell me below."
        ),
        "expression": "guilty and exposed, sheepish grin",
    },
]


def build_car_image_prompt(character, color, expression):
    """Pixar Cars-style character prompt."""
    return (
        f"A Pixar Cars style 3D animated {character} character, front three-quarter view, "
        f"facing camera directly. {color.capitalize()} metallic paint, expressive cartoon eyes "
        f"on the windshield with eyelids and eyebrows showing {expression}. "
        f"Wide talking mouth on the lower bumper/grille with clear lip definition. "
        f"Simple clean garage background with soft warm lighting. "
        f"Portrait composition, centered, vertical 9:16 aspect ratio. "
        f"Highly detailed 3D render, Pixar-quality, cinematic lighting. "
        f"The mouth MUST be prominent with clear lip definition for animation."
    )


def build_object_image_prompt(character, color, expression):
    """Pixar-style inanimate object character prompt."""
    return (
        f"A Pixar style 3D animated {character}, front view, facing camera directly. "
        f"{color.capitalize()} color scheme, expressive cartoon eyes with eyelids and eyebrows "
        f"showing {expression}. Wide talking mouth with clear lip definition. "
        f"The object is anthropomorphized with arms/hands for gesturing. "
        f"Simple clean dark background with dramatic spotlight lighting. "
        f"Portrait composition, centered, vertical 9:16 aspect ratio. "
        f"Highly detailed 3D render, Pixar-quality, cinematic lighting. "
        f"The mouth MUST be prominent with clear lip definition for animation."
    )


def generate_all_images(segments, brand, run_name):
    """Generate character images for all segments. Returns list of image paths."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate {len(segments)} character images")
    print(f"{'='*60}")

    prompt_fn = build_car_image_prompt if brand == "car" else build_object_image_prompt
    image_paths = []

    for i, seg in enumerate(segments):
        prompt = prompt_fn(seg["character"], seg["color"], seg["expression"])
        out_name = f"{run_name}_seg{i+1}_{seg['character'].lower().replace(' ', '_')}"
        print(f"\n[{i+1}/{len(segments)}] Generating: {seg['character']}")
        img_path = generate_image(prompt, out_name, model=IMAGE_MODEL)
        if not img_path:
            raise RuntimeError(f"Image generation failed for segment {i+1}: {seg['character']}")
        image_paths.append(img_path)
        print(f"  -> {img_path}")

    return image_paths


def generate_all_audio(segments, run_name):
    """Generate TTS audio for all segments. Returns list of audio paths."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Generate {len(segments)} TTS audio files")
    print(f"{'='*60}")

    audio_paths = []
    for i, seg in enumerate(segments):
        audio_path = str(OUT / f"{run_name}_seg{i+1}_audio.mp3")
        audio_paths.append(audio_path)

    async def _generate_all():
        for i, seg in enumerate(segments):
            print(f"\n[{i+1}/{len(segments)}] TTS: {seg['character']} ({seg['voice']})")
            comm = edge_tts.Communicate(seg["script"], seg["voice"], rate=TTS_RATE)
            await comm.save(audio_paths[i])
            print(f"  -> {audio_paths[i]}")

    asyncio.run(_generate_all())
    return audio_paths


def lipsync_all(image_paths, audio_paths, segments, run_name):
    """Run HeyGen lip-sync for all segments. Returns list of raw video paths."""
    print(f"\n{'='*60}")
    print(f"PHASE 3: HeyGen lip-sync ({len(segments)} segments)")
    print(f"  Estimated cost: ~${sum(0.10 * 7 for _ in segments):.2f}")
    print(f"{'='*60}")

    raw_paths = []
    for i, (img, aud, seg) in enumerate(zip(image_paths, audio_paths, segments)):
        print(f"\n[{i+1}/{len(segments)}] Lip-sync: {seg['character']}")
        print(f"  Uploading image...")
        image_url = fal_client.upload_file(img)
        print(f"  Uploading audio...")
        audio_url = fal_client.upload_file(aud)
        print(f"  Running HeyGen...")

        result = fal_client.subscribe(
            HEYGEN_MODEL,
            arguments={"image_url": image_url, "audio_url": audio_url},
            with_logs=True,
        )

        video_url = result["video"]["url"]
        raw_path = str(OUT / f"{run_name}_seg{i+1}_raw.mp4")
        video_data = requests.get(video_url).content
        Path(raw_path).write_bytes(video_data)
        print(f"  -> {raw_path} ({len(video_data)/1024:.0f} KB)")
        raw_paths.append(raw_path)

    return raw_paths


def process_all_segments(raw_paths, segments, run_name):
    """Crop, upscale, add captions to each segment. Returns list of processed paths."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: Post-process {len(segments)} segments")
    print(f"{'='*60}")

    processed_paths = []
    for i, (raw, seg) in enumerate(zip(raw_paths, segments)):
        print(f"\n[{i+1}/{len(segments)}] Processing: {seg['character']}")

        clip = VideoFileClip(raw)
        print(f"  Raw: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s")

        # Crop to 9:16
        clip = crop_to_vertical(clip)
        # Upscale to 1080x1920
        clip = clip.resized((1080, 1920))
        # Add captions
        clip = add_captions(clip, seg["script"])

        processed_path = str(OUT / f"{run_name}_seg{i+1}_processed.mp4")
        clip.write_videofile(
            processed_path, codec="libx264", audio_codec="aac",
            fps=30, bitrate="10M", preset="fast", logger=None,
        )
        clip.close()
        print(f"  -> {processed_path}")
        processed_paths.append(processed_path)

    return processed_paths


def stitch_segments(processed_paths, run_name):
    """Concatenate processed segments into one video (ffmpeg concat, robust)."""
    print(f"\n{'='*60}")
    print(f"PHASE 5: Stitch {len(processed_paths)} segments")
    print(f"{'='*60}")

    durations = []
    for p in processed_paths:
        c = VideoFileClip(p)
        durations.append(c.duration)
        c.close()
    total_dur = sum(durations)
    print(f"  Total duration: {total_dur:.1f}s")

    ffmpeg = get_ffmpeg_binary()
    stitched_path = str(OUT / f"{run_name}_stitched.mp4")
    concat_file = OUT / f"{run_name}_concat.txt"

    lines = [f"file '{Path(p).resolve().as_posix()}'" for p in processed_paths]
    concat_file.write_text("\n".join(lines), encoding="ascii")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        stitched_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not Path(stitched_path).exists():
        raise RuntimeError(f"Stitch failed: {result.stderr[:500]}")

    print(f"  -> {stitched_path}")
    return stitched_path


def add_audio_stack(stitched_path, processed_paths, run_name, bgm_path=None, enable_sfx=True):
    """Apply default audio stack: BGM rotation + SFX timing map."""
    print(f"\n{'='*60}")
    print(f"PHASE 6: Add BGM + SFX")
    print(f"{'='*60}")

    final_path = str(OUT / f"{run_name}_final.mp4")
    ffmpeg = get_ffmpeg_binary()

    # Pick BGM (v2 pool first)
    if bgm_path:
        bgm = Path(bgm_path)
    else:
        bgm = pick_existing(BGM_POOL) or pick_existing(BGM_FALLBACKS)

    if not bgm or not bgm.exists():
        print("  No BGM found, copying stitched video as final")
        shutil.copy2(stitched_path, final_path)
        return final_path

    print(f"  BGM: {bgm}")

    # Build segment timing map
    durations = []
    for p in processed_paths:
        c = VideoFileClip(p)
        durations.append(c.duration)
        c.close()

    starts = [0.0]
    for d in durations[:-1]:
        starts.append(starts[-1] + d)

    # SFX events (lite profile): sparse and intentional
    # Default target: ~2 events per 20-30s video
    sfx_events = []

    if enable_sfx and SFX_USE_INTRO:
        intro = pick_existing(SFX_INTRO_POOL)
        if intro:
            sfx_events.append({"path": intro, "time": 0.0, "volume": SFX_INTRO_VOLUME, "label": "intro"})

    if enable_sfx:
        # Transition SFX: max 1 (first segment change)
        transition_times = starts[1:] if len(starts) > 1 else []
        for i, t in enumerate(transition_times[:SFX_MAX_TRANSITIONS], start=1):
            sw = pick_existing(SFX_TRANSITION_POOL)
            if sw:
                sfx_events.append({
                    "path": sw,
                    "time": t,
                    "volume": SFX_TRANSITION_VOLUME,
                    "label": f"transition_{i}",
                })

        # Impact SFX: max 1 at ~60% of full video for emphasis
        total_dur = sum(durations)
        impact_times = [total_dur * 0.60] if total_dur > 3 else []
        for i, t in enumerate(impact_times[:SFX_MAX_IMPACTS], start=1):
            imp = pick_existing(SFX_IMPACT_POOL)
            if imp:
                sfx_events.append({
                    "path": imp,
                    "time": t,
                    "volume": SFX_IMPACT_VOLUME,
                    "label": f"impact_{i}",
                })

    print(f"  SFX events: {len(sfx_events)}")

    # Build ffmpeg input list
    cmd = [ffmpeg, "-y", "-i", stitched_path, "-stream_loop", "-1", "-i", str(bgm)]
    for ev in sfx_events:
        cmd.extend(["-i", str(ev["path"])])

    # Build filter graph
    filters = [f"[1:a]volume={BGM_VOLUME}[bgm]"]
    mix_inputs = ["[0:a]", "[bgm]"]

    for idx, ev in enumerate(sfx_events, start=2):
        delay_ms = int(ev["time"] * 1000)
        label = f"sfx{idx}"
        filters.append(
            f"[{idx}:a]volume={ev['volume']},adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_inputs.append(f"[{label}]")

    amix_inputs = len(mix_inputs)
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={amix_inputs}:duration=first:dropout_transition=0,alimiter=limit=0.95[out]"
    )

    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_path,
    ])

    print("  Running ffmpeg audio stack...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not Path(final_path).exists():
        print(f"  ffmpeg error: {result.stderr[:500]}")
        print("  Falling back to stitched video (no stack)")
        shutil.copy2(stitched_path, final_path)
    else:
        print(f"  -> {final_path}")

    return final_path


def run_pipeline(brand="car", run_name=None, skip_images=False, image_dir=None, bgm_path=None, enable_sfx=True):
    """Full listicle pipeline: images → audio → lip-sync → process → stitch → music → QA."""
    segments = CAR_SEGMENTS if brand == "car" else MOATIFI_SEGMENTS
    title = "4 Cars That Will Bankrupt You After 100K Miles" if brand == "car" else \
            "4 Stocks That Look Cheap But Are Actually Traps"

    if not run_name:
        run_name = f"{brand}_listicle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'#'*60}")
    print(f"LISTICLE VIDEO PIPELINE")
    print(f"Brand: {brand}")
    print(f"Title: {title}")
    print(f"Segments: {len(segments)}")
    print(f"Run: {run_name}")
    print(f"{'#'*60}")

    # Phase 1: Images
    if skip_images and image_dir:
        print(f"\nSkipping image generation, using images from {image_dir}")
        image_paths = sorted(Path(image_dir).glob(f"{run_name}_seg*"))
        image_paths = [str(p) for p in image_paths]
    else:
        image_paths = generate_all_images(segments, brand, run_name)

    # Phase 2: Audio
    audio_paths = generate_all_audio(segments, run_name)

    # Phase 3: Lip-sync
    raw_paths = lipsync_all(image_paths, audio_paths, segments, run_name)

    # Phase 4: Process segments
    processed_paths = process_all_segments(raw_paths, segments, run_name)

    # Phase 5: Stitch
    stitched_path = stitch_segments(processed_paths, run_name)

    # Phase 6: BGM + SFX audio stack
    final_path = add_audio_stack(
        stitched_path,
        processed_paths,
        run_name,
        bgm_path=bgm_path,
        enable_sfx=enable_sfx,
    )

    # Phase 7: QA
    print(f"\n{'='*60}")
    print(f"PHASE 7: QA")
    print(f"{'='*60}")
    from video_qa import run_qa
    qa = run_qa(final_path)

    # Summary
    file_size = Path(final_path).stat().st_size / (1024 * 1024)
    print(f"\n{'#'*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Final: {final_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  QA: {'PASS' if qa['passed'] else 'FAIL'}")
    print(f"{'#'*60}")

    return {
        "final_path": final_path,
        "stitched_path": stitched_path,
        "image_paths": image_paths,
        "audio_paths": audio_paths,
        "raw_paths": raw_paths,
        "processed_paths": processed_paths,
        "qa": qa,
        "title": title,
        "brand": brand,
        "run_name": run_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-character listicle video pipeline")
    parser.add_argument("--brand", default="car", choices=["car", "moatifi"],
                        help="Brand/niche (car or moatifi)")
    parser.add_argument("--name", default=None, help="Run name prefix")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image generation (reuse existing)")
    parser.add_argument("--bgm", default=None,
                        help="Optional path to a specific BGM track")
    parser.add_argument("--no-sfx", action="store_true",
                        help="Disable SFX timing map")
    args = parser.parse_args()

    result = run_pipeline(
        brand=args.brand,
        run_name=args.name,
        skip_images=args.skip_images,
        bgm_path=args.bgm,
        enable_sfx=not args.no_sfx,
    )
