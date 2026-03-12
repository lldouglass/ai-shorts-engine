"""Automatic video QA — runs after every render, before Distribution send."""
import os
import json
import numpy as np
from pathlib import Path
from moviepy import VideoFileClip
from PIL import Image


def extract_qa_frames(video_path, interval=1.0):
    """Extract frames at regular intervals, return as list of numpy arrays + paths."""
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    w, h = clip.size
    fps = clip.fps

    frames = []
    t = 0.0
    while t < duration:
        frame = clip.get_frame(t)
        frames.append({"time": t, "frame": frame})
        t += interval

    clip.close()
    return {
        "frames": frames,
        "width": w,
        "height": h,
        "fps": fps,
        "duration": duration,
    }


def check_resolution(width, height, min_w=1080, min_h=1920):
    """Check if resolution meets delivery minimum."""
    passed = width >= min_w and height >= min_h
    return {
        "check": "resolution",
        "passed": passed,
        "detail": f"{width}x{height} (min {min_w}x{min_h})",
    }


def check_fps(fps, target=30.0, tolerance=1.0):
    """Check if FPS meets target."""
    passed = abs(fps - target) <= tolerance
    return {
        "check": "fps",
        "passed": passed,
        "detail": f"{fps:.2f} (target {target})",
    }


def check_bitrate(video_path, min_mbps=8.0, max_mbps=15.0):
    """Check if bitrate is in acceptable range."""
    size_bytes = Path(video_path).stat().st_size
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    clip.close()
    bitrate = (size_bytes * 8.0 / max(duration, 0.001)) / 1_000_000
    passed = min_mbps <= bitrate <= max_mbps
    return {
        "check": "bitrate",
        "passed": passed,
        "detail": f"{bitrate:.2f} Mbps (target {min_mbps}-{max_mbps})",
        "value": bitrate,
    }


def check_lip_sync(frames_data):
    """Detect if there's actual mouth/face movement between frames.
    
    Compares pixel differences in the center face region across frames.
    Low variance = static/frozen = lip-sync likely failed.
    """
    frames = frames_data["frames"]
    w = frames_data["width"]
    h = frames_data["height"]

    if len(frames) < 3:
        return {
            "check": "lip_sync_movement",
            "passed": False,
            "detail": "Not enough frames to analyze",
        }

    # Focus on center-face region (middle 40% width, 25-55% height for vertical video)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.25)
    y2 = int(h * 0.55)

    face_crops = []
    for f in frames:
        crop = f["frame"][y1:y2, x1:x2]
        face_crops.append(crop.astype(np.float32))

    # Compute frame-to-frame differences in the face region
    diffs = []
    for i in range(1, len(face_crops)):
        diff = np.mean(np.abs(face_crops[i] - face_crops[i - 1]))
        diffs.append(diff)

    avg_diff = np.mean(diffs) if diffs else 0.0
    max_diff = np.max(diffs) if diffs else 0.0

    # Calibrated thresholds (Mar 6 2026):
    #   Good lip-sync (candle): avg=14-15, std=6-9
    #   No lip-sync (car):      avg=4,     std=3
    # Require BOTH avg > 7.0 AND std > 4.0 to pass
    std_diff = float(np.std(diffs)) if diffs else 0.0
    passed = avg_diff > 7.0 and std_diff > 4.0
    
    return {
        "check": "lip_sync_movement",
        "passed": passed,
        "detail": f"avg={avg_diff:.2f}, std={std_diff:.2f}, max={max_diff:.2f} (need avg>7 AND std>4)",
        "avg_diff": float(avg_diff),
        "std_diff": float(std_diff),
        "max_diff": float(max_diff),
    }


def check_captions_visible(frames_data):
    """Check if caption text is present in the lower portion of frames."""
    frames = frames_data["frames"]
    w = frames_data["width"]
    h = frames_data["height"]

    # Check bottom 30% of frame for white text pixels
    caption_detected_count = 0
    for f in frames:
        bottom = f["frame"][int(h * 0.65):int(h * 0.90), :]
        # White-ish pixels (R>200, G>200, B>200) indicate caption text
        white_mask = np.all(bottom > 200, axis=2)
        white_ratio = np.sum(white_mask) / white_mask.size
        if white_ratio > 0.005:  # At least 0.5% white pixels = captions present
            caption_detected_count += 1

    ratio = caption_detected_count / max(len(frames), 1)
    passed = ratio > 0.5  # Captions should be in at least half the frames

    return {
        "check": "captions_visible",
        "passed": passed,
        "detail": f"Detected in {caption_detected_count}/{len(frames)} frames ({ratio:.0%})",
    }


def run_qa(video_path):
    """Run full QA suite on a finished video. Returns pass/fail + details."""
    video_path = str(video_path)
    print(f"\n{'='*50}")
    print(f"QA: {video_path}")
    print(f"{'='*50}")

    # Extract frames
    frames_data = extract_qa_frames(video_path, interval=0.5)

    # Run all checks
    results = []
    results.append(check_resolution(frames_data["width"], frames_data["height"]))
    results.append(check_fps(frames_data["fps"]))
    results.append(check_bitrate(video_path))
    results.append(check_lip_sync(frames_data))
    results.append(check_captions_visible(frames_data))

    # Summary
    all_passed = all(r["passed"] for r in results)
    
    print("\nQA Results:")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['detail']}")

    if all_passed:
        print("\n[PASS] ALL CHECKS PASSED -- ready for Distribution")
    else:
        failed = [r["check"] for r in results if not r["passed"]]
        print(f"\n[FAIL] FAILED CHECKS: {', '.join(failed)} -- DO NOT send to Distribution")

    return {
        "passed": all_passed,
        "results": results,
        "video_path": video_path,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/candle_videos/moatifi_simple_investing_raw_final.mp4"
    qa = run_qa(path)
    if not qa["passed"]:
        exit(1)
