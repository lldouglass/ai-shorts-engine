#!/usr/bin/env python3
"""Test script to verify all API connections are working.

Run: python scripts/test_api_connections.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx

results = {}

def header(t):
    print(f"\n{'='*60}\n {t}\n{'='*60}")

def result(name, ok, msg=""):
    s = "[OK]" if ok else "[X] "
    print(f"  {s} {name}")
    if msg: print(f"      {msg}")
    results[name] = {"ok": ok}

async def test_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key or "..." in key:
        result("OpenAI", False, "API key not configured")
        return
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"}, timeout=10)
            if r.status_code == 200:
                result("OpenAI", True, "Connected")
            else:
                result("OpenAI", False, f"HTTP {r.status_code}")
    except Exception as e:
        result("OpenAI", False, str(e)[:50])

async def test_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        result("Anthropic", False, "Not configured (optional)")
        return
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "Hi"}]}, timeout=30)
            result("Anthropic", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        result("Anthropic", False, str(e)[:50])

async def test_luma():
    key = os.getenv("LUMA_API_KEY")
    if not key or "..." in key:
        result("Luma AI", False, "Not configured")
        return
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://api.lumalabs.ai/dream-machine/v1/generations",
                headers={"Authorization": f"Bearer {key}"}, timeout=10)
            result("Luma AI", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        result("Luma AI", False, str(e)[:50])

async def test_creatomate():
    key = os.getenv("CREATOMATE_API_KEY")
    if not key:
        result("Creatomate", False, "Not configured")
        return
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://api.creatomate.com/v1/templates",
                headers={"Authorization": f"Bearer {key}"}, timeout=10)
            result("Creatomate", r.status_code == 200)
    except Exception as e:
        result("Creatomate", False, str(e)[:50])

async def test_elevenlabs():
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        result("ElevenLabs", False, "Not configured (Edge TTS is free)")
        return
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": key}, timeout=10)
            result("ElevenLabs", r.status_code == 200)
    except Exception as e:
        result("ElevenLabs", False, str(e)[:50])

async def test_edge_tts():
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        result("Edge TTS", True, f"{len(voices)} voices (FREE)")
    except ImportError:
        result("Edge TTS", False, "pip install edge-tts")
    except Exception as e:
        result("Edge TTS", False, str(e)[:50])

def test_oauth(name, *keys):
    if all(os.getenv(k) for k in keys):
        result(name, True, "Configured")
    else:
        result(name, False, "Not configured")

async def test_db():
    url = os.getenv("DATABASE_URL", "postgresql://shorts:shorts@localhost:5432/shorts")
    try:
        from sqlalchemy import create_engine, text
        e = create_engine(url)
        with e.connect() as c:
            v = c.execute(text("SELECT version()")).scalar()
            result("PostgreSQL", True, v[:35]+"...")
    except Exception as e:
        result("PostgreSQL", False, str(e)[:50])

async def test_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        c = redis.from_url(url)
        c.ping()
        result("Redis", True, f"v{c.info()['redis_version']}")
    except Exception as e:
        result("Redis", False, str(e)[:50])

def test_encryption():
    key = os.getenv("ENCRYPTION_MASTER_KEY")
    if not key:
        result("Encryption Key", False, "Not set")
        return
    try:
        from cryptography.fernet import Fernet
        Fernet(key.encode())
        result("Encryption Key", True, "Valid")
    except:
        result("Encryption Key", False, "Invalid")

async def main():
    print("\n" + "="*60 + "\n AI Shorts Engine - API Tests\n" + "="*60)
    
    header("Infrastructure")
    await test_db()
    await test_redis()
    test_encryption()
    
    header("LLM Providers")
    await test_openai()
    await test_anthropic()
    
    header("Video/Rendering")
    await test_luma()
    await test_creatomate()
    
    header("Voiceover")
    await test_elevenlabs()
    await test_edge_tts()
    
    header("Publishing OAuth")
    test_oauth("YouTube", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET")
    test_oauth("TikTok", "TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET")
    test_oauth("Instagram", "INSTAGRAM_APP_ID", "INSTAGRAM_APP_SECRET")
    
    header("Summary")
    p = sum(1 for r in results.values() if r["ok"])
    print(f"\n  {p}/{len(results)} passed\n")

if __name__ == "__main__":
    asyncio.run(main())
