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
- **Multi-Platform Publishing**: YouTube Shorts with multi-account support
  - Connect multiple YouTube accounts via OAuth
  - Schedule uploads with `publishAt`
  - Dry-run mode for testing without uploading
  - Rate limiting (configurable max uploads per day)
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

### Publish Pipeline

The publish pipeline uploads rendered videos to platforms:

```
job_id -> [publish_to_youtube] -> [record_publish_result]
```

1. **publish_to_youtube**: Uploads video via YouTube Data API with metadata and scheduling
2. **record_publish_result**: Stores platform video ID, URL, and any API-forced visibility changes

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

### Video Publishing

```bash
# Connect a YouTube account (interactive OAuth flow)
shorts-engine accounts connect youtube --label "Main Channel"

# List connected accounts
shorts-engine accounts list

# Publish a rendered video to YouTube
shorts-engine shorts publish \
  --job <job-uuid> \
  --youtube-account "Main Channel" \
  --wait

# Schedule a video for later
shorts-engine shorts publish \
  --job <job-uuid> \
  --youtube-account "Main Channel" \
  --publish-at "2024-12-25T10:00:00Z"

# Dry-run mode (shows what would be uploaded without actually uploading)
shorts-engine shorts publish \
  --job <job-uuid> \
  --youtube-account "Main Channel" \
  --dry-run

# Publish as unlisted
shorts-engine shorts publish \
  --job <job-uuid> \
  --youtube-account "Main Channel" \
  --visibility unlisted
```

### Account Management

```bash
# Connect a YouTube account
shorts-engine accounts connect youtube --label "My Channel"

# Use browser-based OAuth instead of device flow
shorts-engine accounts connect youtube --label "My Channel" --browser

# List all connected accounts
shorts-engine accounts list

# List only YouTube accounts
shorts-engine accounts list --platform youtube

# Disconnect an account
shorts-engine accounts disconnect "My Channel"

# Link an account to a project (for publishing)
shorts-engine accounts link --account "Main Channel" --project <project-uuid> --default
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

### Publishing Tables

- **platform_accounts**: Connected YouTube/TikTok/Instagram accounts with encrypted tokens
- **account_projects**: Mapping of which accounts can publish to which projects
- **publish_jobs**: Individual publish operations with status, platform video ID, and URL

### Relationships

```
projects
  └── video_jobs (1:N)
        ├── scenes (1:N)
        │     ├── assets (1:N)
        │     └── prompts (1:N)
        ├── assets (1:N)  # job-level assets
        └── publish_jobs (1:N)

platform_accounts
  ├── account_projects (1:N) -> projects
  └── publish_jobs (1:N)
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
| `YOUTUBE_CLIENT_ID` | YouTube OAuth client ID | - |
| `YOUTUBE_CLIENT_SECRET` | YouTube OAuth client secret | - |
| `YOUTUBE_REDIRECT_URI` | OAuth redirect URI | `http://localhost:8085/callback` |
| `YOUTUBE_MAX_UPLOADS_PER_DAY` | Max uploads per account per day | `50` |
| `ENCRYPTION_MASTER_KEY` | Fernet key for token encryption | (auto-generated in dev) |

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
| `/api/v1/jobs/shorts/publish` | POST | Publish video to platforms |
| `/api/v1/jobs/publish/{id}` | GET | Get publish job status |
| `/api/v1/accounts` | GET | List connected accounts |
| `/api/v1/accounts/{id}` | GET | Get account details |
| `/api/v1/accounts/link` | POST | Link account to project |

## YouTube OAuth Setup

### Prerequisites

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Go to "APIs & Services" > "Credentials"
5. Create OAuth 2.0 credentials (Desktop app type)
6. Download the credentials and note the Client ID and Client Secret

### Configuration

Add these to your `.env` file:

```bash
YOUTUBE_CLIENT_ID=your_client_id.apps.googleusercontent.com
YOUTUBE_CLIENT_SECRET=your_client_secret

# Optional: Generate a secure encryption key for production
# python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
ENCRYPTION_MASTER_KEY=your_generated_fernet_key
```

### Connecting an Account

```bash
# This opens a browser/shows a code for authorization
shorts-engine accounts connect youtube --label "My Channel"

# Follow the prompts to authorize the app
# The refresh token will be securely stored in the database
```

### Troubleshooting

**"No refresh token received"**
- This happens if the app was previously authorized. Go to [Google Account Permissions](https://myaccount.google.com/permissions), find "AI Shorts Engine", and revoke access. Then try connecting again.

**"invalid_grant" error**
- The refresh token has been revoked or expired. Disconnect and reconnect the account:
  ```bash
  shorts-engine accounts disconnect "My Channel"
  shorts-engine accounts connect youtube --label "My Channel"
  ```

**"quotaExceeded" error**
- You've hit YouTube's API quota limit. Wait 24 hours or request a quota increase in Google Cloud Console.

**Video forced to private**
- YouTube may force videos to private if your channel is new or unverified. Complete channel verification to unlock public uploads for longer videos.

**Daily upload limit reached**
- The engine enforces a configurable daily limit (default 50) to prevent API abuse. Adjust with `YOUTUBE_MAX_UPLOADS_PER_DAY` or wait until midnight UTC.

## License

Proprietary - All rights reserved.
