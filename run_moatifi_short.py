"""Moatifi listicle — ultra-tight 3-5 second cuts."""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["FAL_KEY"] = "b34297c9-7603-427d-9c72-c0f3b90890b3:6b567a922a508fa64fe55562f6daae62"

from dotenv import load_dotenv
load_dotenv()

from make_listicle_video import (
    generate_all_images, generate_all_audio, lipsync_all,
    process_all_segments, OUT, BGM_PATH
)
from pathlib import Path
from moviepy import AudioFileClip
import subprocess

# ~10-12 words per segment = 3-5 seconds at -5% rate
SEGMENTS = [
    {
        "character": "a cracked golden trophy with a fake smile",
        "color": "gold",
        "voice": "en-US-AndrewNeural",
        "script": "My PE looks cheap but earnings are crashing. That's a value trap.",
        "expression": "smug but crumbling, cracks showing",
    },
    {
        "character": "a piggy bank covered in bandages",
        "color": "pink",
        "voice": "en-GB-RyanNeural",
        "script": "They bought me for the dividend. Then it got slashed sixty percent.",
        "expression": "injured and annoyed, one eye squinting",
    },
    {
        "character": "a melting ice cube in a tiny business suit",
        "color": "light blue",
        "voice": "en-US-GuyNeural",
        "script": "Up three hundred percent on vibes alone. Now I'm down eighty.",
        "expression": "panicking, sweating, eyes wide",
    },
    {
        "character": "a rusty old safe that's wide open and empty inside",
        "color": "rusty brown",
        "voice": "en-US-ChristopherNeural",
        "script": "Fortress balance sheet? Debt was hidden off the books. Which stock fooled you?",
        "expression": "sheepish guilty grin, caught red-handed",
    },
]

RUN = "moatifi_short_v2"

print(f"{'#'*60}")
print(f"MOATIFI ULTRA-SHORT LISTICLE")
print(f"Target: 3-5s per segment, 12-20s total")
print(f"{'#'*60}")

# Phase 1: Images
image_paths = generate_all_images(SEGMENTS, "moatifi", RUN)

# Phase 2: Audio
audio_paths = generate_all_audio(SEGMENTS, RUN)
total_dur = 0
for i, p in enumerate(audio_paths):
    c = AudioFileClip(p)
    print(f"  Seg {i+1}: {c.duration:.1f}s")
    total_dur += c.duration
    c.close()
print(f"  TOTAL: {total_dur:.1f}s")

# Phase 3: Lip-sync
raw_paths = lipsync_all(image_paths, audio_paths, SEGMENTS, RUN)

# Phase 4: Process each segment
processed_paths = process_all_segments(raw_paths, SEGMENTS, RUN)

# Phase 5+6: Stitch + music via ffmpeg (skip moviepy concat)
ffmpeg = r"C:\Users\Logan\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
concat_file = str(OUT / "concat_short.txt")
lines = [f"file '{p.replace(os.sep, '/')}'" for p in processed_paths]
with open(concat_file, "w", encoding="ascii") as f:
    f.write("\n".join(lines))

stitched = str(OUT / f"{RUN}_stitched.mp4")
subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c", "copy", stitched], capture_output=True)
print(f"Stitched: {stitched}")

final = str(OUT / f"{RUN}_final.mp4")
bgm = str(BGM_PATH)
subprocess.run([ffmpeg, "-y", "-i", stitched, "-stream_loop", "-1", "-i", bgm,
                "-filter_complex", "[1:a]volume=0.15[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[out]",
                "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest", final], capture_output=True)

size = Path(final).stat().st_size / (1024*1024)
print(f"Final: {final} ({size:.1f} MB)")
print(f"Est cost: ${total_dur * 0.10:.2f}")

# Telegram preview
preview = r"C:\Users\Logan\.openclaw\workspace\moatifi_short_preview.mp4"
subprocess.run([ffmpeg, "-y", "-i", final, "-c:v", "libx264", "-b:v", "2500k",
                "-c:a", "aac", "-b:a", "128k", preview], capture_output=True)
psize = Path(preview).stat().st_size / (1024*1024)
print(f"Preview: {preview} ({psize:.1f} MB)")
print(f"\n{'#'*60}")
print(f"DONE — {total_dur:.1f}s total, ${total_dur * 0.10:.2f} HeyGen cost")
print(f"{'#'*60}")
