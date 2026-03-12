from pathlib import Path
from make_listicle_video import add_audio_stack

stitched = 'output/listicle_videos/moatifi_short_v2_stitched.mp4'
processed = [
    'output/listicle_videos/moatifi_short_v2_seg1_processed.mp4',
    'output/listicle_videos/moatifi_short_v2_seg2_processed.mp4',
    'output/listicle_videos/moatifi_short_v2_seg3_processed.mp4',
    'output/listicle_videos/moatifi_short_v2_seg4_processed.mp4',
]
out = add_audio_stack(
    stitched,
    processed,
    run_name='audio_stack_smoketest_lite',
    bgm_path='music_v2/bgm_v2_driving_ambition.mp3',
    enable_sfx=True,
)
print('OUT', out, Path(out).exists(), Path(out).stat().st_size)
