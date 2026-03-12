"""Simple ffmpeg-based renderer. Concat clips + overlay voiceover.

Way more reliable than MoviePy for basic composition.
Uses imageio-ffmpeg's bundled ffmpeg binary.
"""

import subprocess
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_ffmpeg() -> str:
    """Get ffmpeg binary path from imageio-ffmpeg."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_duration(ffmpeg: str, path: Path) -> float:
    """Get duration of a media file using ffprobe."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not Path(ffprobe).exists():
        # Fallback: use ffmpeg -i
        result = subprocess.run(
            [ffmpeg, "-i", str(path)],
            capture_output=True, text=True
        )
        # Parse duration from stderr
        for line in result.stderr.split("\n"):
            if "Duration:" in line:
                time_str = line.split("Duration:")[1].split(",")[0].strip()
                parts = time_str.split(":")
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return 0.0
    
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def render(clips_dir: Path, voiceover_path: Path, output_path: Path) -> Path:
    """Concat video clips and overlay voiceover audio."""
    ffmpeg = get_ffmpeg()
    
    # Find all scene clips in order
    clips = sorted(clips_dir.glob("scene_*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No scene clips found in {clips_dir}")
    
    print(f"Found {len(clips)} clips")
    
    # Get durations
    total_video = 0.0
    for clip in clips:
        dur = get_duration(ffmpeg, clip)
        total_video += dur
        print(f"  {clip.name}: {dur:.1f}s ({clip.stat().st_size:,} bytes)")
    
    vo_duration = get_duration(ffmpeg, voiceover_path) if voiceover_path.exists() else 0.0
    print(f"\nVideo total: {total_video:.1f}s")
    print(f"Voiceover: {vo_duration:.1f}s")
    
    # Step 1: Create concat list
    concat_file = clips_dir.parent / "concat.txt"
    with open(concat_file, "w") as f:
        for clip in clips:
            # Escape backslashes for ffmpeg
            escaped = str(clip.absolute()).replace("\\", "/")
            f.write(f"file '{escaped}'\n")
    
    # Step 2: Concat clips into one video (no audio)
    concat_path = clips_dir.parent / "concat_video.mp4"
    print(f"\nConcatenating clips...")
    
    subprocess.run([
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",  # No re-encoding for concat
        str(concat_path)
    ], check=True, capture_output=True)
    
    concat_dur = get_duration(ffmpeg, concat_path)
    print(f"Concat video: {concat_dur:.1f}s")
    
    # Step 3: If voiceover is longer than video, extend video by looping last clip
    if vo_duration > concat_dur + 1.0:
        print(f"\nVoiceover is {vo_duration - concat_dur:.1f}s longer than video.")
        print("Extending video with last clip loop...")
        
        extended_path = clips_dir.parent / "extended_video.mp4"
        
        # Re-encode with tpad to freeze last frame
        subprocess.run([
            ffmpeg, "-y",
            "-i", str(concat_path),
            "-vf", f"tpad=stop_mode=clone:stop_duration={vo_duration - concat_dur + 0.5}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(extended_path)
        ], check=True, capture_output=True)
        
        concat_path = extended_path
        concat_dur = get_duration(ffmpeg, concat_path)
        print(f"Extended video: {concat_dur:.1f}s")
    
    # Step 4: Merge video + voiceover audio
    print(f"\nMerging video + voiceover...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if voiceover_path.exists() and vo_duration > 0:
        # Use shortest to avoid hanging if durations don't match perfectly
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
    else:
        # No voiceover, just copy
        import shutil
        shutil.copy2(concat_path, output_path)
    
    final_dur = get_duration(ffmpeg, output_path)
    final_size = output_path.stat().st_size
    
    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"Output: {output_path}")
    print(f"Duration: {final_dur:.1f}s")
    print(f"Size: {final_size / 1024 / 1024:.1f}MB")
    print(f"{'='*50}")
    
    # Cleanup temp files
    for tmp in [clips_dir.parent / "concat.txt", 
                clips_dir.parent / "concat_video.mp4",
                clips_dir.parent / "extended_video.mp4"]:
        if tmp.exists() and tmp != output_path:
            tmp.unlink(missing_ok=True)
    
    return output_path


if __name__ == "__main__":
    base = Path(__file__).parent / "output" / "viral_cat"
    render(
        clips_dir=base / "clips",
        voiceover_path=base / "voiceover.mp3",
        output_path=base / "final_video.mp4",
    )
