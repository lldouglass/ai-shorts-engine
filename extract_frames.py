"""Extract key frames from a video for visual QA."""
import sys
import os
from moviepy import VideoFileClip

def extract_frames(video_path, output_dir=None, interval=2.0):
    """Extract frames every `interval` seconds."""
    if output_dir is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{base}_frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    clip = VideoFileClip(video_path)
    duration = clip.duration
    print(f"Video: {video_path}")
    print(f"Duration: {duration:.1f}s, Size: {clip.size}, FPS: {clip.fps}")
    
    t = 0.0
    frame_num = 0
    while t < duration:
        frame = clip.get_frame(t)
        # Save as PNG
        from PIL import Image
        img = Image.fromarray(frame)
        out_path = os.path.join(output_dir, f"frame_{frame_num:03d}_{t:.1f}s.png")
        img.save(out_path)
        print(f"  Saved: {out_path}")
        frame_num += 1
        t += interval
    
    clip.close()
    print(f"\nExtracted {frame_num} frames to {output_dir}")
    return output_dir

if __name__ == "__main__":
    video = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    extract_frames(video, interval=interval)
