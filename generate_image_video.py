"""Image-based video pipeline.

Generates story → images (Gemini 2.5 Flash) → voiceover (Edge TTS) → 
Ken Burns render (ffmpeg). Target 61+ seconds for TikTok Creator Rewards.

Cost: ~$0.05-0.25/video (Google credits) or free on free tier.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import edge_tts
import openai
from google import genai
from google.genai import types as genai_types


# ── Config ──────────────────────────────────────────────────────────
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "imagen-4.0-ultra-generate-001")
STORY_MODEL = os.getenv("STORY_MODEL", "gpt-4o-mini")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewNeural")  # Good dramatic male voice
TARGET_DURATION = 65  # seconds, >60 for TikTok Creator Rewards
IMAGES_PER_VIDEO = 8
SECONDS_PER_IMAGE = TARGET_DURATION / IMAGES_PER_VIDEO  # ~8.1s each
OUTPUT_DIR = Path(__file__).parent / "output"
WIDTH, HEIGHT = 1080, 1920  # 9:16 vertical


def get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── Step 1: Story Generation ───────────────────────────────────────
def generate_story(topic: str) -> dict:
    """Generate a story with scene-by-scene image prompts."""
    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model=STORY_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": f"""You are a viral short-form video story writer.

Create a {IMAGES_PER_VIDEO}-scene visual story for a 70-second narrated video.
The story must have: a hook in scene 1, escalation, a twist, and a satisfying ending.

Return JSON:
{{
    "title": "Video title",
    "narration": "Full narration script, 180-200 words. Written for voiceover - conversational, engaging, with natural pauses. MUST be at least 180 words to fill 65+ seconds of audio.",
    "scenes": [
        {{
            "scene_number": 1,
            "image_prompt": "Detailed image generation prompt. Include: subject, action, setting, lighting, style. Always specify 'vertical portrait orientation, 9:16 aspect ratio'. Be very specific about character appearance for consistency.",
            "caption": "Short scene description (3-5 words)"
        }}
    ],
    "character_description": "Detailed physical description of the main character that should be consistent across ALL scenes. Include: species/type, color, distinguishing features, accessories, expression style.",
    "hashtags": ["relevant", "trending", "hashtags"]
}}

IMPORTANT: Every image_prompt must include the exact character_description to maintain visual consistency across scenes."""},
            {"role": "user", "content": f"Create a viral video story about: {topic}"}
        ],
    )
    
    return json.loads(response.choices[0].message.content)


# ── Step 2: Image Generation ──────────────────────────────────────
def generate_images(story: dict, output_dir: Path) -> list[Path]:
    """Generate images using best available model (Imagen 4 Ultra or Gemini fallback)."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    char_desc = story.get("character_description", "")
    is_imagen = "imagen" in IMAGE_MODEL.lower()
    paths = []
    
    for scene in story["scenes"]:
        scene_num = scene["scene_number"]
        prompt = scene["image_prompt"]
        
        # Prepend character description for consistency
        if char_desc and char_desc.lower() not in prompt.lower():
            prompt = f"Character: {char_desc}. Scene: {prompt}"
        
        print(f"   Scene {scene_num}/{IMAGES_PER_VIDEO}: {scene['caption']}")
        
        for attempt in range(3):
            try:
                if is_imagen:
                    # Imagen 4 API (generate_images)
                    response = client.models.generate_images(
                        model=IMAGE_MODEL,
                        prompt=prompt,
                        config=genai_types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio="9:16",
                        ),
                    )
                    img_bytes = response.generated_images[0].image.image_bytes
                else:
                    # Gemini multimodal API (generate_content)
                    response = client.models.generate_content(
                        model=IMAGE_MODEL,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )
                    img_bytes = None
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                            img_bytes = part.inline_data.data
                            break
                    if not img_bytes:
                        print(f"   [WARN] No image in response, retrying...")
                        continue
                
                img_path = images_dir / f"scene_{scene_num:02d}.png"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"   [OK] {len(img_bytes):,} bytes")
                paths.append(img_path)
                break  # Success
                
            except Exception as e:
                err = str(e)[:150]
                print(f"   [FAIL] Attempt {attempt+1}: {err}")
                if "RESOURCE_EXHAUSTED" in str(e):
                    print("   [WAIT] Rate limited, waiting 30s...")
                    time.sleep(30)
                elif attempt < 2:
                    time.sleep(5)
        
        # Rate limit pause
        if scene_num < IMAGES_PER_VIDEO:
            time.sleep(3)
    
    return paths


# ── Step 3: Voiceover Generation ──────────────────────────────────
async def generate_voiceover(text: str, output_path: Path) -> Path:
    """Generate voiceover using Edge TTS (free)."""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(str(output_path))
    return output_path


# ── Step 4: Ken Burns Render ──────────────────────────────────────
def get_duration(ffmpeg: str, path: Path) -> float:
    """Get media file duration."""
    result = subprocess.run(
        [ffmpeg, "-i", str(path)],
        capture_output=True, text=True
    )
    for line in result.stderr.split("\n"):
        if "Duration:" in line:
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            parts = time_str.split(":")
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def render_ken_burns(images: list[Path], voiceover_path: Path, output_path: Path) -> Path:
    """Render images with Ken Burns effects + voiceover."""
    ffmpeg = get_ffmpeg()
    
    vo_duration = get_duration(ffmpeg, voiceover_path)
    secs_per_image = max(vo_duration / len(images), 6.0)
    total_frames = int(secs_per_image * 30)  # 30fps
    crossfade = 1.0  # 1 second crossfade
    
    print(f"   Voiceover: {vo_duration:.1f}s")
    print(f"   Images: {len(images)}")
    print(f"   Per image: {secs_per_image:.1f}s ({total_frames} frames)")
    
    temp_dir = output_path.parent / "temp_clips"
    temp_dir.mkdir(exist_ok=True)
    clip_paths = []
    
    # Ken Burns effects: alternate between zoom-in, zoom-out, pan-left, pan-right
    effects = [
        # Slow zoom in (1.0 → 1.15)
        f"zoompan=z='min(zoom+0.0005,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={WIDTH}x{HEIGHT}:fps=30",
        # Slow zoom out (1.15 → 1.0)
        f"zoompan=z='if(eq(on,1),1.15,max(zoom-0.0005,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={WIDTH}x{HEIGHT}:fps=30",
        # Pan right
        f"zoompan=z='1.1':x='if(eq(on,1),0,min(x+1,iw-iw/zoom))':y='ih/2-(ih/zoom/2)':d={total_frames}:s={WIDTH}x{HEIGHT}:fps=30",
        # Pan left  
        f"zoompan=z='1.1':x='if(eq(on,1),iw-iw/zoom,max(x-1,0))':y='ih/2-(ih/zoom/2)':d={total_frames}:s={WIDTH}x{HEIGHT}:fps=30",
    ]
    
    for i, img_path in enumerate(images):
        print(f"   Ken Burns scene {i+1}/{len(images)}...")
        clip_path = temp_dir / f"clip_{i:02d}.mp4"
        effect = effects[i % len(effects)]
        
        subprocess.run([
            ffmpeg, "-y",
            "-loop", "1", "-i", str(img_path),
            "-vf", effect,
            "-t", str(secs_per_image),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(clip_path)
        ], check=True, capture_output=True)
        
        clip_paths.append(clip_path)
    
    # Concatenate with crossfades
    print(f"   Concatenating with {crossfade}s crossfades...")
    
    if len(clip_paths) == 1:
        concat_path = clip_paths[0]
    else:
        # Build complex filter for xfade transitions
        inputs = []
        for cp in clip_paths:
            inputs.extend(["-i", str(cp)])
        
        # Chain xfade filters
        filter_parts = []
        current_offset = secs_per_image - crossfade
        
        # First transition
        filter_parts.append(
            f"[0][1]xfade=transition=fade:duration={crossfade}:offset={current_offset}[v1]"
        )
        
        for i in range(2, len(clip_paths)):
            current_offset += secs_per_image - crossfade
            prev = f"v{i-1}"
            out = f"v{i}"
            filter_parts.append(
                f"[{prev}][{i}]xfade=transition=fade:duration={crossfade}:offset={current_offset}[{out}]"
            )
        
        last_label = f"v{len(clip_paths)-1}"
        filter_complex = ";".join(filter_parts)
        
        concat_path = temp_dir / "concat.mp4"
        cmd = [ffmpeg, "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{last_label}]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(concat_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: simple concat without crossfades
            print(f"   [WARN] Crossfade failed, using simple concat")
            concat_file = temp_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for cp in clip_paths:
                    f.write(f"file '{str(cp.absolute()).replace(chr(92), '/')}'\n")
            
            subprocess.run([
                ffmpeg, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(concat_path)
            ], check=True, capture_output=True)
    
    # Merge video + voiceover
    print(f"   Merging video + voiceover...")
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(concat_path),
        "-i", str(voiceover_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(output_path)
    ], check=True, capture_output=True)
    
    # Get final info
    final_dur = get_duration(ffmpeg, output_path)
    final_size = output_path.stat().st_size
    
    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path, final_dur, final_size


# ── Main Pipeline ─────────────────────────────────────────────────
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Image-based video pipeline")
    parser.add_argument("--topic", required=True, help="Video topic/concept")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--voice", default=TTS_VOICE, help="Edge TTS voice")
    args = parser.parse_args()
    
    # Setup output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / f"img_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Image-Based Video Pipeline")
    print("=" * 60)
    print(f"  Topic: {args.topic}")
    print(f"  Model: {IMAGE_MODEL}")
    print(f"  Voice: {args.voice}")
    print(f"  Target: {TARGET_DURATION}s")
    print(f"  Output: {out_dir}")
    
    # Step 1: Story
    print(f"\n{'='*60}")
    print("[STORY] Generating story...")
    print(f"{'='*60}")
    story = generate_story(args.topic)
    print(f"   Title: {story['title']}")
    print(f"   Scenes: {len(story['scenes'])}")
    print(f"   Character: {story.get('character_description', 'N/A')[:100]}")
    print(f"   Narration: {story['narration'][:200]}...")
    
    # Save story
    with open(out_dir / "story.json", "w") as f:
        json.dump(story, f, indent=2)
    
    # Step 2: Images
    print(f"\n{'='*60}")
    print(f"[IMAGES] Generating {IMAGES_PER_VIDEO} images with {IMAGE_MODEL}...")
    print(f"{'='*60}")
    images = generate_images(story, out_dir)
    print(f"   Generated: {len(images)}/{IMAGES_PER_VIDEO}")
    
    if len(images) < 3:
        print("[ERROR] Too few images generated. Aborting.")
        return
    
    # Step 3: Voiceover
    print(f"\n{'='*60}")
    print("[VOICE] Generating voiceover (Edge TTS - free)...")
    print(f"{'='*60}")
    vo_path = out_dir / "voiceover.mp3"
    await generate_voiceover(story["narration"], vo_path)
    print(f"   [OK] Saved to {vo_path}")
    
    # Step 4: Render
    print(f"\n{'='*60}")
    print("[RENDER] Ken Burns + transitions...")
    print(f"{'='*60}")
    final_path = out_dir / "final_video.mp4"
    _, duration, size = render_ken_burns(images, vo_path, final_path)
    
    print(f"\n{'='*60}")
    print("  VIDEO COMPLETE!")
    print(f"{'='*60}")
    print(f"   Output: {final_path}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Size: {size / 1024 / 1024:.1f}MB")
    print(f"   Title: {story['title']}")
    print(f"   Hashtags: {' '.join('#' + h for h in story.get('hashtags', []))}")
    
    if duration >= 61:
        print(f"   ✓ TikTok Creator Rewards eligible (>{60}s)")
    else:
        print(f"   ✗ Under 61s - won't qualify for TikTok Creator Rewards")


if __name__ == "__main__":
    asyncio.run(main())
