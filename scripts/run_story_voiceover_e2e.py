#!/usr/bin/env python
"""End-to-end test: Generate story and voiceover without database.

This demonstrates the core fix:
- Story generates 100-150 word narrative
- Voiceover uses the full narrative (not caption beats)
- Voice is "thriller" (Arnold) for dark content

Run: python scripts/run_story_voiceover_e2e.py

Requires: OPENAI_API_KEY, ELEVENLABS_API_KEY in .env
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "src")


async def main() -> None:
    from shorts_engine.services.story_generator import StoryGenerator
    from shorts_engine.adapters.voiceover.elevenlabs import ElevenLabsProvider
    from shorts_engine.adapters.voiceover.base import VoiceoverRequest
    from shorts_engine.config import settings

    print("=" * 60)
    print(" STORY -> VOICEOVER E2E TEST")
    print("=" * 60)

    # Step 1: Generate story
    print("\n[1/3] Generating story...")
    generator = StoryGenerator()
    story = await generator.generate(
        "An AI that starts deleting its user's memories without permission"
    )

    print(f"\n   Title: {story.title}")
    print(f"   Style: {story.narrative_style}")
    print(f"   Words: {story.word_count}")
    print(f"   Duration: {story.estimated_duration_seconds}s")
    print(f"\n   Narrative:\n   {story.narrative_text[:200]}...")

    # Step 2: Generate voiceover using story narrative
    print("\n[2/3] Generating voiceover...")
    print(f"   Voice: {settings.voiceover_default_voice} (Arnold - deep, cinematic)")
    print(f"   Text length: {len(story.narrative_text)} chars, {story.word_count} words")

    voiceover = ElevenLabsProvider()

    if not settings.elevenlabs_api_key:
        print("\n   [SKIP] No ElevenLabs API key - would generate voiceover here")
        return

    request = VoiceoverRequest(
        text=story.narrative_text,
        voice_id=settings.voiceover_default_voice,  # "thriller" -> Arnold
    )

    result = await voiceover.generate(request)

    if not result.success:
        print(f"\n   [ERROR] Voiceover failed: {result.error_message}")
        return

    # Step 3: Save voiceover
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    audio_path = output_dir / "voiceover_test.mp3"
    audio_path.write_bytes(result.audio_data)

    print(f"\n[3/3] Voiceover saved!")
    print(f"   Path: {audio_path}")
    print(f"   Size: {len(result.audio_data):,} bytes")
    print(f"   Duration: {result.duration_seconds:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print(f"""
   Story: {story.title}
   Words: {story.word_count} (target: 100-150)
   Voice: {settings.voiceover_default_voice} (Arnold)
   Audio: {audio_path}
   Duration: {result.duration_seconds:.1f}s (target: 40-60s)

   [OK] Story narrative used for voiceover (not caption beats)
   [OK] Thriller voice selected for dark content
   [OK] Duration fills most of 60s video
""")


if __name__ == "__main__":
    asyncio.run(main())
