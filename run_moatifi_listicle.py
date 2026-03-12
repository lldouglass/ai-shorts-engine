"""Run Moatifi listicle video with short 5-second segments."""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

os.environ["FAL_KEY"] = "b34297c9-7603-427d-9c72-c0f3b90890b3:6b567a922a508fa64fe55562f6daae62"

from dotenv import load_dotenv
load_dotenv()

from make_listicle_video import (
    generate_all_images, generate_all_audio, lipsync_all,
    process_all_segments, stitch_segments, add_background_music,
    build_object_image_prompt, OUT, BGM_PATH, IMAGE_MODEL
)
from video_qa import run_qa
from pathlib import Path
from datetime import datetime

# Short punchy segments — target ~5 seconds each (~15 words)
SEGMENTS = [
    {
        "character": "a cracked golden trophy with a fake smile",
        "color": "gold",
        "voice": "en-US-AndrewNeural",
        "script": (
            "My PE looks cheap but earnings are falling off a cliff. "
            "Low price doesn't mean good deal."
        ),
        "expression": "smug but crumbling, cracks showing",
    },
    {
        "character": "a piggy bank covered in bandages",
        "color": "pink",
        "voice": "en-GB-RyanNeural",
        "script": (
            "They bought me for the fat dividend. Then the company "
            "cut it sixty percent. Always check the payout ratio."
        ),
        "expression": "injured and annoyed, one eye squinting",
    },
    {
        "character": "a melting ice cube in a tiny business suit",
        "color": "light blue",
        "voice": "en-US-GuyNeural",
        "script": (
            "Up three hundred percent in six months. No moat, no profits, "
            "just vibes. Now I'm down eighty percent."
        ),
        "expression": "panicking, sweating, eyes wide",
    },
    {
        "character": "a rusty old safe that's wide open and completely empty inside",
        "color": "rusty brown",
        "voice": "en-US-ChristopherNeural",
        "script": (
            "Fortress balance sheet? My debt was hidden off the books. "
            "What stock fooled you the worst? Tell me below."
        ),
        "expression": "sheepish guilty grin, caught red-handed",
    },
]

TITLE = "4 Stocks That Look Cheap But Are Actually Traps"
RUN_NAME = f"moatifi_traps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"\n{'#'*60}")
print(f"MOATIFI LISTICLE — {TITLE}")
print(f"Target: ~5s per segment, ~20s total")
print(f"Run: {RUN_NAME}")
print(f"{'#'*60}")

# Phase 1: Images
image_paths = generate_all_images(SEGMENTS, "moatifi", RUN_NAME)

# Phase 2: Audio
audio_paths = generate_all_audio(SEGMENTS, RUN_NAME)

# Check durations
from moviepy import AudioFileClip
total_dur = 0
for i, p in enumerate(audio_paths):
    c = AudioFileClip(p)
    print(f"  Seg {i+1} audio: {c.duration:.1f}s")
    total_dur += c.duration
    c.close()
print(f"  Total audio: {total_dur:.1f}s")

# Phase 3: Lip-sync
raw_paths = lipsync_all(image_paths, audio_paths, SEGMENTS, RUN_NAME)

# Phase 4: Process
processed_paths = process_all_segments(raw_paths, SEGMENTS, RUN_NAME)

# Phase 5: Stitch
stitched_path = stitch_segments(processed_paths, RUN_NAME)

# Phase 6: Music
final_path = add_background_music(stitched_path, BGM_PATH, RUN_NAME)

# Phase 7: QA
qa = run_qa(final_path)

size_mb = Path(final_path).stat().st_size / (1024 * 1024)
print(f"\n{'#'*60}")
print(f"DONE: {final_path}")
print(f"Size: {size_mb:.1f} MB")
print(f"Duration: {total_dur:.1f}s estimated")
print(f"QA: {'PASS' if qa['passed'] else 'FAIL'}")
print(f"Est cost: ${total_dur * 0.10:.2f} (HeyGen)")
print(f"{'#'*60}")
