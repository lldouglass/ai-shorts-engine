"""Calibrate lip-sync detection thresholds."""
from video_qa import extract_qa_frames
import numpy as np

videos = {
    "candle_hook": "output/hook_test/heygen_lipsync_test_final.mp4",
    "candle_moatifi": "output/candle_videos/moatifi_simple_investing_raw_final.mp4",
    "car_broken": "output/car_videos/car_listen_to_your_car_final.mp4",
}

for name, path in videos.items():
    data = extract_qa_frames(path, interval=0.5)
    frames = data["frames"]
    w, h = data["width"], data["height"]

    # Face region
    x1, x2 = int(w * 0.30), int(w * 0.70)
    y1, y2 = int(h * 0.25), int(h * 0.55)

    crops = [f["frame"][y1:y2, x1:x2].astype(np.float32) for f in frames]

    diffs = []
    for i in range(1, len(crops)):
        diff = np.mean(np.abs(crops[i] - crops[i - 1]))
        diffs.append(diff)

    avg = np.mean(diffs)
    std = np.std(diffs)
    mx = np.max(diffs)

    print(f"{name:20s}: avg={avg:.2f}, std={std:.2f}, max={mx:.2f}, diffs={[f'{d:.2f}' for d in diffs]}")
