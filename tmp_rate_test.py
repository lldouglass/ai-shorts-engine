import asyncio
import edge_tts
from moviepy import AudioFileClip

text = "I cost a hundred thousand dollars and I scored a two out of five on reliability. My air suspension fails, my electronics glitch, and I have been recalled five times. But hey, at least I look good in the shop."

for rate in ["+30%", "+50%", "+70%", "+85%", "+100%"]:
    out = f"output/car_videos/tmp_rr_{rate.replace('%','').replace('+','')}.mp3"
    async def gen():
        c = edge_tts.Communicate(text, "en-US-AndrewNeural", rate=rate)
        await c.save(out)
    asyncio.run(gen())
    d = AudioFileClip(out).duration
    print(rate, round(d, 2))
