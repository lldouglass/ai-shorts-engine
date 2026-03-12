"""Build final Range Rover clip: no bars, normal-speed voiceover, synced captions."""
import asyncio
from pathlib import Path

import edge_tts
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips

from post_process import add_captions
from video_qa import run_qa

OUT = Path("output/car_videos")
OUT.mkdir(parents=True, exist_ok=True)

RAW_VIDEO = OUT / "veo31_rr_nb_front_raw.mp4"
AUDIO_PATH = OUT / "rr_unreliable_voice_normal.mp3"
TMP_PATH = OUT / "veo31_rr_nb_front_unreliable_normal_tmp.mp4"
FINAL_PATH = OUT / "veo31_rr_nb_front_unreliable_normal_nobars_final.mp4"

SCRIPT = (
    "I cost a hundred thousand dollars and I scored a two out of five on reliability. "
    "My air suspension fails, my electronics glitch, and I have been recalled five times. "
    "But hey, at least I look good in the shop."
)

VOICE = "en-US-AndrewNeural"
RATE = "+0%"


async def gen_tts(text: str, out_path: Path):
    comm = edge_tts.Communicate(text, VOICE, rate=RATE)
    await comm.save(str(out_path))


def main():
    print(f"Generating voiceover ({VOICE}, {RATE})...")
    asyncio.run(gen_tts(SCRIPT, AUDIO_PATH))

    clip = VideoFileClip(str(RAW_VIDEO))
    print(f"Raw: {clip.size} {clip.duration:.2f}s @ {clip.fps:.2f}fps")

    # Remove main top/bottom bars from Veo output.
    clip = clip.cropped(y1=104, y2=1177)

    # Resize to delivery vertical.
    clip = clip.resized((1080, 1920))

    audio = AudioFileClip(str(AUDIO_PATH))
    print(f"Audio duration: {audio.duration:.2f}s")

    if audio.duration > clip.duration:
        extra = audio.duration - clip.duration
        last = clip.get_frame(max(0.0, clip.duration - 1.0 / max(1.0, clip.fps)))
        hold = ImageClip(last).with_duration(extra)
        clip = concatenate_videoclips([clip, hold], method="compose")
        print(f"Extended video by {extra:.2f}s")
    else:
        clip = clip.subclipped(0, audio.duration)

    clip = clip.with_audio(audio)
    clip = add_captions(clip, SCRIPT)
    clip = clip.with_fps(30)

    print(f"Rendering temp: {TMP_PATH}")
    clip.write_videofile(
        str(TMP_PATH),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        bitrate="10M",
        preset="slow",
        logger=None,
    )

    # Micro-trim to remove any residual edge bars.
    clip2 = VideoFileClip(str(TMP_PATH)).cropped(y1=4, y2=1916).resized((1080, 1920)).with_fps(30)

    print(f"Rendering final: {FINAL_PATH}")
    clip2.write_videofile(
        str(FINAL_PATH),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        bitrate="10M",
        preset="slow",
        logger=None,
    )

    print("Done")
    run_qa(str(FINAL_PATH))


if __name__ == "__main__":
    main()
