"""Tooth video #2 — standing on tongue, new dental tip."""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.environ["FAL_KEY"] = "b34297c9-7603-427d-9c72-c0f3b90890b3:6b567a922a508fa64fe55562f6daae62"

from dotenv import load_dotenv
load_dotenv()

import fal_client, requests, asyncio, subprocess
import edge_tts
from pathlib import Path
from gemini_image import generate_image
from post_process import crop_to_vertical, add_captions
from moviepy import VideoFileClip, AudioFileClip

OUT = Path("output/tooth_video")
OUT.mkdir(parents=True, exist_ok=True)

SCRIPT = (
    "You're brushing way too hard. Soft bristles only. "
    "Hard brushing wears down your enamel and makes your gums recede. "
    "Gentle circles, two minutes, twice a day. That's it."
)

IMAGE_PROMPT = (
    "A Pixar style 3D animated happy white tooth character standing on top of a big pink cartoon tongue "
    "inside a mouth. The tooth is front view, facing camera directly. "
    "Bright white and shiny with a big friendly confident smile showing clear lip definition and a wide talking mouth. "
    "Expressive cartoon eyes with eyelids, looking wise and helpful like a friendly teacher. "
    "Tiny cartoon arms and hands, one hand pointing up like making a point. "
    "The background is the inside of a cartoon mouth with pink gums and other teeth visible behind, "
    "soft pink and red tones. The tongue is large, pink, and the tooth is standing on it like a stage. "
    "Portrait composition, centered, vertical 9:16 aspect ratio. "
    "Highly detailed 3D render, Pixar-quality, cinematic lighting, warm tones. "
    "The mouth MUST be prominent with clear lip definition for animation."
)

VOICE = "en-US-AndrewNeural"
RUN = "tooth_soft_bristles"

print(f"{'#'*60}")
print(f"TOOTH VIDEO #2 — Soft Bristles Only")
print(f"{'#'*60}")

# Image
print("\n--- Generate character ---")
img_path = generate_image(IMAGE_PROMPT, RUN, model="nano-banana-pro-preview")
print(f"Image: {img_path}")

# TTS
print("\n--- Voiceover ---")
audio_path = str(OUT / f"{RUN}_audio.mp3")
async def _tts():
    comm = edge_tts.Communicate(SCRIPT, VOICE, rate="-5%")
    await comm.save(audio_path)
asyncio.run(_tts())
ac = AudioFileClip(audio_path)
print(f"Audio: {ac.duration:.1f}s")
ac.close()

# HeyGen
print("\n--- HeyGen lip-sync ---")
image_url = fal_client.upload_file(img_path)
audio_url = fal_client.upload_file(audio_path)
result = fal_client.subscribe(
    "fal-ai/heygen/avatar4/image-to-video",
    arguments={"image_url": image_url, "audio_url": audio_url},
    with_logs=True,
)
raw_path = str(OUT / f"{RUN}_raw.mp4")
vdata = requests.get(result["video"]["url"]).content
Path(raw_path).write_bytes(vdata)
print(f"Raw: {raw_path} ({len(vdata)//1024} KB)")

# Post-process
print("\n--- Post-process ---")
clip = VideoFileClip(raw_path)
print(f"Raw: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s")
clip = crop_to_vertical(clip)
clip = clip.resized((1080, 1920))
clip = add_captions(clip, SCRIPT)
processed = str(OUT / f"{RUN}_processed.mp4")
clip.write_videofile(processed, codec="libx264", audio_codec="aac",
                     fps=30, bitrate="10M", preset="fast", logger=None)
clip.close()

# Music
print("\n--- Add music ---")
ffmpeg = r"C:\Users\Logan\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
final = str(OUT / f"{RUN}_final.mp4")
subprocess.run([ffmpeg, "-y", "-i", processed, "-stream_loop", "-1",
                "-i", "music/bgm_chill.mp3",
                "-filter_complex", "[1:a]volume=0.15[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[out]",
                "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest", final], capture_output=True)

size = Path(final).stat().st_size / (1024*1024)
print(f"Final: {final} ({size:.1f} MB)")

# Preview
preview = r"C:\Users\Logan\.openclaw\workspace\tooth_v2_preview.mp4"
subprocess.run([ffmpeg, "-y", "-i", final, "-c:v", "libx264", "-b:v", "2500k",
                "-c:a", "aac", "-b:a", "128k", preview], capture_output=True)
psize = Path(preview).stat().st_size / (1024*1024)
dur = VideoFileClip(final).duration
print(f"Preview: {psize:.1f} MB")
print(f"\n{'#'*60}")
print(f"DONE — {dur:.1f}s, ${dur * 0.10:.2f} cost")
print(f"{'#'*60}")
