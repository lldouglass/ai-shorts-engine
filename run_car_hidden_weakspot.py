import os
import subprocess
from pathlib import Path

from make_listicle_video import (
    generate_all_audio,
    lipsync_all,
    process_all_segments,
    stitch_segments,
    add_audio_stack,
)

RUN = "car_hidden_weakspots_v1"

SEGMENTS = [
    {
        "character": "Toyota RAV4",
        "voice": "en-US-AndrewNeural",
        "script": "RAV4 is solid, but some hybrid years can get high-voltage cable corrosion. That repair can get expensive fast.",
    },
    {
        "character": "Honda Accord",
        "voice": "en-GB-RyanNeural",
        "script": "Accord is reliable overall, but some one-point-five turbo years had head gasket and fuel dilution complaints.",
    },
    {
        "character": "Subaru Forester",
        "voice": "en-US-GuyNeural",
        "script": "Forester is loved, but some CVT years had valve body failures. Which reliable car burned you worst?",
    },
]

IMAGE_PATHS = [
    "output/car_videos/car_hidden_weakspot_rav4.jpg",
    "output/car_videos/car_hidden_weakspot_accord.jpg",
    "output/car_videos/car_hidden_weakspot_forester.jpg",
]


def main():
    for p in IMAGE_PATHS:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing image: {p}")

    print("=== Phase 1: TTS ===")
    audio_paths = generate_all_audio(SEGMENTS, RUN)

    print("=== Phase 2: Lip-sync ===")
    raw_paths = lipsync_all(IMAGE_PATHS, audio_paths, SEGMENTS, RUN)

    print("=== Phase 3: Process ===")
    processed_paths = process_all_segments(raw_paths, SEGMENTS, RUN)

    print("=== Phase 4: Stitch ===")
    stitched_path = stitch_segments(processed_paths, RUN)

    print("=== Phase 5: Audio stack (BGM v2 + lite SFX) ===")
    final_path = add_audio_stack(
        stitched_path,
        processed_paths,
        RUN,
        bgm_path="music_v2/bgm_v2_driving_ambition.mp3",
        enable_sfx=True,
    )

    ffmpeg = "C:\\Users\\Logan\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\imageio_ffmpeg\\binaries\\ffmpeg-win-x86_64-v7.1.exe"
    preview = "C:\\Users\\Logan\\.openclaw\\workspace\\car_hidden_weakspots_preview.mp4"
    subprocess.run([
        ffmpeg, "-y", "-i", final_path,
        "-c:v", "libx264", "-b:v", "2500k",
        "-c:a", "aac", "-b:a", "128k",
        preview,
    ], check=False)

    size_mb = Path(final_path).stat().st_size / (1024 * 1024)
    print(f"FINAL: {final_path} ({size_mb:.1f} MB)")
    print(f"PREVIEW: {preview}")


if __name__ == "__main__":
    main()
