import asyncio
import edge_tts

async def main():
    c = edge_tts.Communicate(
        "These reliable cars each hide one expensive flaw.",
        "en-US-AndrewNeural",
        rate="-5%",
    )
    await c.save("output/listicle_videos/hook_line_option1.mp3")

asyncio.run(main())
