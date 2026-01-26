# AI Shorts Engine

A closed-loop AI Shorts engine that generates 60-second videos, uploads them to platforms, ingests performance metrics, and iterates based on feedback.

## Features

- **Video Generation Pipeline**: End-to-end pipeline from idea to render-ready video
  - LLM-powered scene planning (OpenAI GPT-4 or Anthropic Claude)
  - Pluggable video generation providers (Luma AI, stub for testing)
  - Automatic scene clip generation with style consistency
  - Asset verification and management
- **Render Pipeline**: Compose final videos with voiceover and captions
  - Creatomate integration for professional video composition
  - Voiceover generation (ElevenLabs or Edge TTS fallback)
  - Burned-in captions from caption beats
  - Optional background music mixing
  - 9:16 vertical output (1080x1920) for Shorts/Reels/TikTok
- **Style Presets**: Pre-configured visual styles for different content types
  - DARK_DYSTOPIAN_ANIME: Gritty, cinematic anime with fog and ink linework
  - VIBRANT_MOTION_GRAPHICS: Bold, colorful motion graphics
  - CINEMATIC_REALISM: Photorealistic cinematic footage
  - SURREAL_DREAMSCAPE: Ethereal, dream-like visuals
- **Multi-Platform Publishing**: Adapters for YouTube Shorts, TikTok, Instagram Reels
- **Analytics Ingestion**: Track views, engagement, and performance metrics
- **Comment Analysis**: Ingest and analyze audience feedback
- **Closed-Loop Iteration**: Use performance data to improve future content

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design.

### Video Creation Pipeline

The pipeline processes video creation in stages:

```
idea + preset -> [plan_job] -> [generate_scene_clips] -> [verify_assets] -> [mark_ready_for_render]
```

1. **plan_job**: LLM generates title, description, and 7-8 scenes with visual prompts
2. **generate_scene_clips**: Video provider generates clips for each scene (parallel)
3. **verify_assets**: Validates all assets are ready and accessible
4. **mark_ready_for_render**: Marks job complete with scene clips ready

### Render Pipeline

The render pipeline composes scene clips into a final publishable video:

```
job_id -> [generate_voiceover] -> [render_final_video] -> [mark_ready_to_publish]
```

1. **generate_voiceover** (optional): Creates narration audio from caption beats using ElevenLabs or Edge TTS
2. **render_final_video**: Uses Creatomate to compose clips, burn in captions, and mix audio
3. **mark_ready_to_publish**: Updates job with final MP4 URL

All stages support:
- Retries with exponential backoff
- Idempotency keys to prevent duplicate work on re-runs
- Proper error handling and status tracking

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Make

### Local Development with Docker

```bash
# Copy environment file
cp .env.example .env

# Start all services
make up

# Run migrations
make migrate

# Check health
curl http://localhost:8000/health

# Trigger smoke test job
curl -X POST http://localhost:8000/api/v1/jobs/smoke
```

### Local Development without Docker

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Start Postgres and Redis (required)
docker compose up -d postgres redis

# Run migrations
alembic upgrade head

# Start API server
uvicorn shorts_engine.main:app --reload

# Start Celery worker (in another terminal)
celery -A shorts_engine.worker worker --loglevel=info
```

## CLI Usage

### Video Creation

```bash
# List available style presets
shorts-engine presets

# Create a project (content brand/channel)
shorts-engine projects create --name "Dark Anime Channel" --preset DARK_DYSTOPIAN_ANIME

# List projects
shorts-engine projects list

# Create a short video
shorts-engine shorts create \
  --project <project-uuid> \
  --idea "A lone samurai walks through a fog-covered ruined city at dawn, searching for his lost companion" \
  --preset DARK_DYSTOPIAN_ANIME

# Create and wait for completion
shorts-engine shorts create \
  --project <project-uuid> \
  --idea "Your video concept here" \
  --preset CINEMATIC_REALISM \
  --wait

# Check pipeline status
shorts-engine shorts status <task-id>

# View job details
shorts-engine shorts job <job-id>

# List recent jobs
shorts-engine shorts list --project <project-uuid>
```

### Video Rendering

```bash
# Render a completed job into final MP4
shorts-engine shorts render --job <job-uuid> --wait

# Render without voiceover
shorts-engine shorts render --job <job-uuid> --no-voiceover

# Render with custom voice and background music
shorts-engine shorts render \
  --job <job-uuid> \
  --voice narrator \
  --music https://example.com/music.mp3 \
  --wait
```

### Other Commands

```bash
# Show available commands
shorts-engine --help

# Run smoke test
shorts-engine smoke

# Legacy video generation
shorts-engine generate --topic "AI trends"

# Check job status
shorts-engine status <job-id>

# Check service health
shorts-engine health

# Start Celery worker (development)
shorts-engine worker
```

## Project Structure

```
ai-shorts-engine/
├── src/shorts_engine/
│   ├── api/              # FastAPI routes
│   ├── adapters/         # External service adapters
│   │   ├── llm/          # LLM providers (OpenAI, Anthropic, stub)
│   │   ├── video_gen/    # Video generation (Luma, stub)
│   │   ├── renderer/     # Video rendering (Creatomate, stub)
│   │   ├── voiceover/    # Voice generation (ElevenLabs, Edge TTS, stub)
│   │   ├── publisher/    # Platform publishers
│   │   ├── analytics/    # Analytics adapters
│   │   └── comments/     # Comment ingestion
│   ├── domain/           # Domain models and business logic
│   ├── db/               # Database models and sessions
│   ├── jobs/             # Celery task definitions
│   │   ├── tasks.py      # Core tasks
│   │   ├── video_pipeline.py  # Video creation pipeline
│   │   └── render_pipeline.py # Render pipeline tasks
│   ├── presets/          # Style preset definitions
│   └── services/         # Application services
│       ├── planner.py    # LLM-powered video planning
│       ├── storage.py    # Asset storage management
│       └── pipeline.py   # Pipeline orchestration
├── migrations/           # Alembic database migrations
├── tests/               # Test suite
└── docker-compose.yml
```

## Database Schema

### Pipeline Tables

- **projects**: Content brands/channels
- **video_jobs**: Individual video generation jobs
- **scenes**: Per-job scene definitions (ordered)
- **assets**: Generated clips, audio, final videos
- **prompts**: Exact prompts used for each scene

### Relationships

```
projects
  └── video_jobs (1:N)
        ├── scenes (1:N)
        │     ├── assets (1:N)
        │     └── prompts (1:N)
        └── assets (1:N)  # job-level assets
```

## Development

```bash
# Run linter
make lint

# Run formatter
make format

# Run tests
make test

# Run all checks
make check
```

## Configuration

All configuration is done via environment variables. See `.env.example` for available options.

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `LLM_PROVIDER` | LLM for planning (openai, anthropic, stub) | `openai` |
| `VIDEO_GEN_PROVIDER` | Video generation (luma, stub) | `stub` |
| `RENDERER_PROVIDER` | Video rendering (creatomate, stub) | `stub` |
| `VOICEOVER_PROVIDER` | Voice generation (elevenlabs, edge_tts, stub) | `stub` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `LUMA_API_KEY` | Luma AI API key | - |
| `CREATOMATE_API_KEY` | Creatomate API key | - |
| `CREATOMATE_WEBHOOK_URL` | Webhook for render completion | - |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | - |

## Style Presets

### DARK_DYSTOPIAN_ANIME
Gritty, cinematic anime style with fog, debris, and ink linework. Perfect for dramatic, post-apocalyptic narratives.

- Aspect Ratio: 9:16 (vertical)
- Scene Duration: 5 seconds
- Camera Style: Slow dramatic pans, close-ups with depth of field

### VIBRANT_MOTION_GRAPHICS
Bold, colorful motion graphics with geometric shapes and smooth transitions. Ideal for explainer videos and upbeat content.

- Aspect Ratio: 9:16 (vertical)
- Scene Duration: 4 seconds
- Camera Style: Smooth zooms, seamless shape morphing

### CINEMATIC_REALISM
Photorealistic cinematic footage with film grain and dramatic lighting. Great for storytelling and immersive content.

- Aspect Ratio: 9:16 (vertical)
- Scene Duration: 6 seconds
- Camera Style: Cinematic tracking shots, smooth dollys, epic reveals

### SURREAL_DREAMSCAPE
Ethereal, dream-like visuals with impossible geometry and soft glow. Perfect for artistic and abstract content.

- Aspect Ratio: 9:16 (vertical)
- Scene Duration: 5.5 seconds
- Camera Style: Floating camera, impossible angles, smooth morphing transitions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/ready` | GET | Readiness with component status |
| `/api/v1/jobs/smoke` | POST | Trigger smoke test |
| `/api/v1/jobs/{id}` | GET | Get job status |
| `/api/v1/jobs/shorts/create` | POST | Start video creation pipeline |
| `/api/v1/jobs/shorts/render` | POST | Start render pipeline |

## License

Proprietary - All rights reserved.
