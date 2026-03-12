"""Post-process HeyGen output: crop to 9:16, upscale to 1080x1920, add captions."""
import sys
from pathlib import Path
from moviepy import VideoFileClip


def crop_to_vertical(clip):
    """Center-crop a landscape clip to 9:16 vertical."""
    w, h = clip.size
    # Target: 9:16 ratio from current height
    target_w = int(h * 9 / 16)
    target_h = h
    
    if target_w > w:
        # If the crop would be wider than the video, crop by width instead
        target_w = w
        target_h = int(w * 16 / 9)
    
    x_center = w / 2
    y_center = h / 2
    
    cropped = clip.cropped(
        x1=int(x_center - target_w / 2),
        y1=int(y_center - target_h / 2),
        x2=int(x_center + target_w / 2),
        y2=int(y_center + target_h / 2)
    )
    print(f"Cropped: {w}x{h} -> {target_w}x{target_h}")
    return cropped


def add_captions(clip, text, font_size=None):
    """Add captions via Pillow frame-by-frame for pixel-perfect positioning."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    w, h = clip.size
    duration = clip.duration
    
    # Split text into chunks of ~4-5 words
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= 4:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    time_per_chunk = duration / len(chunks)
    
    # Scale font with output height so style stays consistent across resolutions
    if font_size is None:
        font_size = max(28, int(h * 0.039))  # 28 @ 720p tall, ~75 @ 1920p tall

    # Load font
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    font = ImageFont.truetype(font_path, font_size)
    
    # Pre-render caption images (RGBA with transparency)
    caption_images = []
    for chunk in chunks:
        # Create a transparent overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Wrap text to fit within 80% of frame width
        max_width = int(w * 0.80)
        lines = []
        current_line = []
        for word in chunk.split():
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > max_width and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))
        
        # Calculate total text block height
        line_height = font_size + 6
        total_text_height = len(lines) * line_height
        
        # Position: bottom of text block at 82% of frame height
        y_bottom = int(h * 0.82)
        y_start = y_bottom - total_text_height
        
        # Draw each line centered with stroke
        for j, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            x = (w - text_w) // 2
            y = y_start + j * line_height
            
            # Black stroke (draw text offset in 8 directions)
            stroke = 2
            for dx in [-stroke, 0, stroke]:
                for dy in [-stroke, 0, stroke]:
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
            
            # White text on top
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        
        caption_images.append(np.array(overlay))
    
    # Apply captions frame-by-frame
    original_get_frame = clip.get_frame
    
    def make_frame(t):
        frame = original_get_frame(t)
        
        # Determine which caption to show
        chunk_idx = min(int(t / time_per_chunk), len(caption_images) - 1)
        overlay = caption_images[chunk_idx]
        
        # Composite: blend where overlay alpha > 0
        alpha = overlay[:, :, 3:4] / 255.0
        rgb = overlay[:, :, :3]
        
        # Ensure dimensions match
        fh, fw = frame.shape[:2]
        oh, ow = rgb.shape[:2]
        if fh != oh or fw != ow:
            return frame
        
        blended = (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)
        return blended
    
    from moviepy import VideoClip
    new_clip = VideoClip(make_frame, duration=duration)
    new_clip = new_clip.with_fps(clip.fps)
    if clip.audio:
        new_clip = new_clip.with_audio(clip.audio)
    
    return new_clip


def post_process(video_path, caption_text, output_path=None):
    """Full post-processing pipeline: crop + upscale + captions + high-quality export."""
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_final.mp4"

    print(f"Loading: {video_path}")
    clip = VideoFileClip(str(video_path))
    print(f"Original: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s @ {clip.fps:.2f}fps")

    # Step 1: Crop to vertical
    clip = crop_to_vertical(clip)

    # Step 2: Upscale to delivery minimum 1080x1920
    clip = clip.resized((1080, 1920))
    print("Upscaled to: 1080x1920")

    # Step 3: Add captions (after upscale for crisp text)
    if caption_text:
        clip = add_captions(clip, caption_text)

    # Step 4: Normalize to 30fps target
    clip = clip.with_fps(30)

    # Step 5: High bitrate export (target 8-12 Mbps)
    print(f"Rendering to: {output_path}")
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        bitrate="10M",
        preset="slow",
        logger=None,
    )
    print(f"Done! {output_path}")
    return str(output_path)


if __name__ == "__main__":
    video = sys.argv[1]
    caption = sys.argv[2] if len(sys.argv) > 2 else "Most investors lose money because they're trying to be smart. The real edge? Just avoid being stupid."
    post_process(video, caption)
