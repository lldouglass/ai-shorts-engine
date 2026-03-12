"""Single talking tooth video — spit don't rinse."""
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
from moviepy import VideoFileClip

OUT = Path("output/tooth_video")
OUT.mkdir(parents=True, exist_ok=True)

SCRIPT = (
    "Stop rinsing after you brush! You're washing away all the fluoride. "
    "Just spit and leave it. Your enamel needs that protection sitting on your teeth. "
    "Rinse with water and you just wasted the whole point of brushing."
)

IMAGE_PROMPT = (
    "A Pixar style 3D animated happy white tooth character, front view, facing camera directly. "
    "Bright white and shiny with a big friendly smile showing clear lip definition and a wide talking mouth. "
    "Expressive cartoon eyes with eyelids, looking slightly annoyed but helpful, like a teacher correcting a common mistake. "
    "Tiny cartoon arms and hands gesturing. "
    "Dark navy blue gradient background filling the entire frame, no white edges. "
    "Portrait composition, centered, vertical 9:16 aspect ratio. "
    "Highly detailed 3D render, Pixar-quality, cinematic lighting, soft shadows. "
    "The mouth MUST be prominent with clear lip definition for animation."
)

VOICE = "en-US-AndrewNeural"
TTS_RATE = "-5%"
RUN = "tooth_spit_not_rinse"

print(f"{'#'*60}")
print(f"TOOTH VIDEO — Spit Don't Rinse")
print(f"{'#'*60}")

# Phase 1: Image
print("\n--- Generate tooth character ---")
img_path = generate_image(IMAGE_PROMPT, RUN, model="nano-banana-pro-preview")
print(f"Image: {img_path}")

# Phase 2: TTS
print("\n--- Generate voiceover ---")
audio_path = str(OUT / f"{RUN}_audio.mp3")
async def _tts():
    comm = edge_tts.Communicate(SCRIPT, VOICE, rate=TTS_RATE)
    await comm.save(audio_path)
asyncio.run(_tts())
from moviepy import AudioFileClip
ac = AudioFileClip(audio_path)
print(f"Audio: {ac.duration:.1f}s")
ac.close()

# Phase 3: HeyGen lip-sync
print("\n--- HeyGen lip-sync ---")
image_url = fal_client.upload_file(img_path)
audio_url = fal_client.upload_file(audio_path)
result = fal_client.subscribe(
    "fal-ai/heygen/avatar4/image-to-video",
    arguments={"image_url": image_url, "audio_url": audio_url},
    with_logs=True,
)
raw_path = str(OUT / f"{RUN}_raw.mp4")
video_data = requests.get(result["video"]["url"]).content
Path(raw_path).write_bytes(video_data)
print(f"Raw: {raw_path} ({len(video_data)//1024} KB)")

# Phase 4: Post-process (crop + upscale + captions)
print("\n--- Post-process ---")
clip = VideoFileClip(raw_path)
print(f"Raw: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s")
clip = crop_to_vertical(clip)
clip = clip.resized((1080, 1920))
clip = add_captions(clip, SCRIPT)
processed_path = str(OUT / f"{RUN}_processed.mp4")
clip.write_videofile(processed_path, codec="libx264", audio_codec="aac",
                     fps=30, bitrate="10M", preset="fast", logger=None)
clip.close()

# Phase 5: Add background music
print("\n--- Add music ---")
ffmpeg = r"C:\Users\Logan\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
bgm = str(Path("music/bgm_energetic.mp3"))
final_path = str(OUT / f"{RUN}_final.mp4")
subprocess.run([ffmpeg, "-y", "-i", processed_path, "-stream_loop", "-1", "-i", bgm,
                "-filter_complex", "[1:a]volume=0.15[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[out]",
                "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest", final_path], capture_output=True)

size = Path(final_path).stat().st_size / (1024*1024)
print(f"Final: {final_path} ({size:.1f} MB)")

# Telegram preview
preview = r"C:\Users\Logan\.openclaw\workspace\tooth_spit_preview.mp4"
subprocess.run([ffmpeg, "-y", "-i", final_path, "-c:v", "libx264", "-b:v", "2500k",
                "-c:a", "aac", "-b:a", "128k", preview], capture_output=True)
psize = Path(preview).stat().st_size / (1024*1024)
print(f"Preview: {preview} ({psize:.1f} MB)")

dur = VideoFileClip(final_path).duration
print(f"\n{'#'*60}")
print(f"DONE — {dur:.1f}s, ${dur * 0.10:.2f} HeyGen cost")
print(f"{'#'*60}")
