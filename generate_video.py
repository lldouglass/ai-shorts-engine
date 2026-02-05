"""Standalone video generation script - Story-First Pipeline.

Generates a complete AI short video using:
- StoryGenerator for story with strong hooks and unique premises
- PlannerService with target_duration to match voiceover length
- Google Veo for video generation
- ElevenLabs for voiceover (using story narrative)
- MoviePy for local rendering
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix Windows encoding for console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment from .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
else:
    print(f"Warning: .env not found at {env_path}")

from shorts_engine.services.story_generator import StoryGenerator, Story
from shorts_engine.services.planner import PlannerService, VideoPlan
from shorts_engine.adapters.video_gen.veo import VeoProvider
from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider
from shorts_engine.adapters.renderer.moviepy_renderer import MoviePyRenderer
from shorts_engine.adapters.renderer.creatomate import (
    CreatomateRenderRequest,
    SceneClip,
)
from shorts_engine.presets.styles import get_preset
from shorts_engine.utils.frame_extraction import extract_last_frame
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


async def generate_story(topic: str) -> Story:
    """Generate a story using StoryGenerator with improved prompts."""
    print("\n" + "=" * 60)
    print("[STORY] Generating story with improved prompts...")
    print("=" * 60)
    print(f"   Topic: {topic}")

    generator = StoryGenerator()
    story = await generator.generate(topic)

    print(f"\n   Title: {story.title}")
    print(f"   Style: {story.narrative_style}")
    print(f"   Preset: {story.suggested_preset}")
    print(f"   Words: {story.word_count}")
    print(f"   Est. Duration: {story.estimated_duration_seconds}s")
    print(f"\n   HOOK (first sentence):")
    first_sentence = story.narrative_text.split(".")[0] + "."
    print(f'   "{first_sentence}"')
    print(f"\n   Full narrative:")
    print(f"   {story.narrative_text}")

    return story


async def generate_plan(story: Story) -> VideoPlan:
    """Generate a video plan using PlannerService with target duration."""
    print("\n" + "=" * 60)
    print("[PLAN] Generating video plan with target duration...")
    print("=" * 60)
    print(f"   Target duration: {story.estimated_duration_seconds}s (from story)")
    print(f"   Style preset: {story.suggested_preset}")

    planner = PlannerService()

    # Build story context
    story_context = {
        "narrative_style": story.narrative_style,
        "topic": story.topic,
    }

    plan = await planner.plan(
        idea=story.narrative_text,
        style_preset_name=story.suggested_preset,
        story_context=story_context,
        target_duration_seconds=story.estimated_duration_seconds,
    )

    print(f"\n   Title: {plan.title}")
    print(f"   Scenes: {len(plan.scenes)}")
    print(f"   Total duration: {plan.total_duration}s")
    print(f"\n   Scene breakdown:")
    for scene in plan.scenes:
        print(f"      {scene.scene_number}. [{scene.duration_seconds}s] {scene.caption_beat}")

    return plan


async def generate_voiceover(story: Story) -> tuple[Path | None, float | None]:
    """Generate voiceover using story narrative (not caption beats)."""
    print("\n" + "=" * 60)
    print("[VOICE] Generating voiceover from story narrative...")
    print("=" * 60)

    from shorts_engine.config import settings
    from shorts_engine.adapters.voiceover.base import VoiceoverRequest

    print(f"   Voice: {settings.voiceover_default_voice}")
    print(f"   Text: {story.word_count} words, {len(story.narrative_text)} chars")

    if not settings.elevenlabs_api_key:
        print("   [SKIP] No ElevenLabs API key")
        return None, None

    elevenlabs = ElevenLabsProvider()
    request = VoiceoverRequest(
        text=story.narrative_text,
        voice_id=settings.voiceover_default_voice,
    )

    result = await elevenlabs.generate(request)

    if not result.success:
        print(f"   [FAIL] Voiceover failed: {result.error_message}")
        return None, None

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    voiceover_path = output_dir / "voiceover.mp3"

    # Write and explicitly close to avoid Windows file locking
    with open(voiceover_path, "wb") as f:
        f.write(result.audio_data)

    # Get ACTUAL duration from the saved file (ElevenLabs estimate is often wrong)
    from moviepy import AudioFileClip

    audio_clip = AudioFileClip(str(voiceover_path))
    actual_duration = audio_clip.duration
    audio_clip.close()

    print(f"   [OK] Saved to {voiceover_path}")
    print(f"   ElevenLabs estimate: {result.duration_seconds}s")
    print(f"   Actual duration: {actual_duration}s")

    return voiceover_path, actual_duration


async def generate_video_clips(plan: VideoPlan) -> list[Path]:
    """Generate video clips using Veo with frame chaining for consistency."""
    print("\n" + "=" * 60)
    print("[VIDEO] Generating video clips with Veo...")
    print("=" * 60)

    from shorts_engine.config import settings

    preset = get_preset(plan.style_preset)
    style_suffix = f", {preset.format_style_prompt()}" if preset else ""
    negative_prompt = preset.format_negative_prompt() if preset else None

    veo = VeoProvider()
    output_dir = Path("output/clips")
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    previous_frame: bytes | None = None
    frame_chaining_enabled = settings.video_frame_chaining_enabled

    if frame_chaining_enabled:
        print(f"   [CHAIN] Frame chaining ENABLED for visual consistency")
    if negative_prompt:
        print(f"   [NEG] Negative prompt: {negative_prompt[:50]}...")

    for scene in plan.scenes:
        print(f"\n   Scene {scene.scene_number}/{len(plan.scenes)}: {scene.caption_beat}")
        print(f"   Duration: {scene.duration_seconds}s")

        prompt = f"{scene.visual_prompt}{style_suffix}"

        from shorts_engine.adapters.video_gen.base import VideoGenRequest

        # Veo 3.1 supports 4, 6, or 8 seconds (handled by VeoProvider)
        clip_duration = int(scene.duration_seconds)

        # Build reference images from previous frame if chaining enabled
        reference_images = None
        if frame_chaining_enabled and previous_frame:
            reference_images = [previous_frame]
            print(f"   [CHAIN] Using previous frame as reference")

        request = VideoGenRequest(
            prompt=prompt,
            duration_seconds=clip_duration,
            aspect_ratio="9:16",
            negative_prompt=negative_prompt,
            reference_images=reference_images,
        )

        result = await veo.generate(request)

        if not result.success:
            print(f"   [FAIL] {result.error_message}")
            # Don't break chain - next scene will generate without reference
            previous_frame = None
            continue

        clip_path = output_dir / f"scene_{scene.scene_number:02d}.mp4"

        video_url = result.metadata.get("video_url") if result.metadata else None
        download_headers = result.metadata.get("download_headers", {}) if result.metadata else {}

        if video_url:
            import httpx

            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                try:
                    response = await client.get(video_url, headers=download_headers)
                    response.raise_for_status()
                    clip_path.write_bytes(response.content)
                    clip_paths.append(clip_path)
                    print(f"   [OK] Saved to {clip_path} ({len(response.content):,} bytes)")

                    # Extract last frame for next scene's reference (during rate limit wait)
                    if frame_chaining_enabled:
                        try:
                            previous_frame = extract_last_frame(clip_path)
                            print(
                                f"   [CHAIN] Extracted last frame ({len(previous_frame):,} bytes)"
                            )
                        except Exception as e:
                            print(f"   [WARN] Frame extraction failed: {e}")
                            previous_frame = None

                except Exception as e:
                    print(f"   [FAIL] Download failed: {e}")
                    previous_frame = None
        else:
            print(f"   [FAIL] No video URL in result")
            previous_frame = None

        # Rate limiting
        if scene.scene_number < len(plan.scenes):
            print(f"   [WAIT] Rate limit pause (15s)...")
            await asyncio.sleep(15)

    return clip_paths


async def render_final_video(
    clip_paths: list[Path],
    plan: VideoPlan,
    voiceover_path: Path | None,
) -> Path:
    """Render final video using MoviePy."""
    print("\n" + "=" * 60)
    print("[RENDER] Rendering final video with MoviePy...")
    print("=" * 60)

    renderer = MoviePyRenderer(output_dir=Path("output"))

    # Build scene clips (only for scenes we have clips for)
    # No caption text - visuals only
    scene_clips = []
    for i, clip_path in enumerate(clip_paths):
        if i < len(plan.scenes):
            scene = plan.scenes[i]
            scene_clips.append(
                SceneClip(
                    video_url=f"file://{clip_path.absolute()}",
                    duration_seconds=scene.duration_seconds,
                    caption_text=None,  # No on-screen text
                    scene_number=scene.scene_number,
                )
            )

    print(f"   Clips: {len(scene_clips)}")
    print(f"   Voiceover: {voiceover_path or 'None'}")

    request = CreatomateRenderRequest(
        scenes=scene_clips,
        voiceover_url=f"file://{voiceover_path.absolute()}" if voiceover_path else None,
        width=1080,
        height=1920,
        fps=30,
    )

    result = await renderer.render_composition(request)

    if not result.success:
        raise RuntimeError(f"Render failed: {result.error_message}")

    print(f"   [OK] Rendered to {result.output_path}")
    print(f"   Duration: {result.duration_seconds}s")

    return result.output_path


async def main():
    """Main entry point - Story-First Pipeline."""
    print("\n" + "=" * 60)
    print("  AI Shorts Engine - Story-First Pipeline")
    print("=" * 60)

    # Configuration - topic for story generation
    # Avoiding topics that trigger Veo content filters (violence, children, crime, death)
    topic = "A smart home AI that starts making decisions for you without asking"

    print(f"\n[TOPIC] {topic}")

    # Check API keys
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
    }

    missing = [k for k, v in required_keys.items() if not v]
    if missing:
        print(f"\n[ERROR] Missing API keys: {', '.join(missing)}")
        print("Please ensure these are set in your .env file")
        return

    print("[OK] All API keys found")

    try:
        # Step 1: Generate story with improved prompts
        story = await generate_story(topic)

        # Step 2: Generate voiceover from story narrative
        voiceover_path, actual_duration = await generate_voiceover(story)

        # Step 3: Generate plan with target duration from voiceover
        # Use actual voiceover duration if available, otherwise use story estimate
        target_duration = actual_duration or story.estimated_duration_seconds

        # Update story's estimated duration if we have actual voiceover duration
        if actual_duration:
            print(f"\n[INFO] Using actual voiceover duration: {actual_duration}s")
            story.estimated_duration_seconds = actual_duration

        plan = await generate_plan(story)

        # Step 4: Generate video clips
        clip_paths = await generate_video_clips(plan)

        if not clip_paths:
            print("\n[ERROR] No video clips were generated")
            return

        # Step 5: Render final video
        final_path = await render_final_video(clip_paths, plan, voiceover_path)

        # Summary
        print("\n" + "=" * 60)
        print("  VIDEO GENERATION COMPLETE!")
        print("=" * 60)
        print(f"""
   OUTPUT: {final_path}

   Story:
      Title: {story.title}
      Hook: "{story.narrative_text.split(".")[0]}."
      Words: {story.word_count}

   Video:
      Scenes: {len(plan.scenes)}
      Duration: {plan.total_duration}s
      Voiceover: {actual_duration}s

   [OK] Video duration matches voiceover length!
   [OK] Full story narrative plays (including twist)
""")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
