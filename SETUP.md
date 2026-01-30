# Get AI Shorts Engine Running

## Prerequisites
- Docker & Docker Compose installed
- Python 3.11+ (for local dev)

## Quick Start Steps

### 1. Create `.env` file in project root
```bash
# Database & Redis (Docker handles these)
DATABASE_URL=postgresql://shorts:shorts@localhost:5432/shorts
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Providers - use "stub" for testing without API keys
LLM_PROVIDER=stub
VIDEO_GEN_PROVIDER=veo
RENDERER_PROVIDER=stub
VOICEOVER_PROVIDER=stub

# Your API keys (already configured)
GOOGLE_API_KEY=<your-key>

# Optional: Add these for full functionality
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# CREATOMATE_API_KEY=
# ELEVENLABS_API_KEY=
```

### 2. Start services
```bash
make up
```
This starts: PostgreSQL, Redis, FastAPI, Celery worker, Celery beat

### 3. Run database migrations
```bash
make migrate
```

### 4. Verify it's working
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

### 5. Run smoke test
```bash
make smoke
```

## Key Services (after `make up`)
| Service | Port | Purpose |
|---------|------|---------|
| API | 8000 | FastAPI server |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Task queue |
| Worker | - | Celery task processor |
| Beat | - | Scheduled tasks |

## Useful Commands
```bash
make logs      # View all logs
make down      # Stop everything
make test      # Run tests
make lint      # Check code
```

## Verification
- `curl http://localhost:8000/health` returns `{"status": "healthy"}`
- `curl http://localhost:8000/health/ready` shows all components ready
- `make smoke` completes successfully


## Never touch the .env file. 