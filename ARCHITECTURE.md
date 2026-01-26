# Architecture

## System Overview

The AI Shorts Engine is a closed-loop content generation system designed to:

1. **Generate** short-form video content using AI
2. **Publish** to multiple platforms (YouTube Shorts, TikTok, Instagram Reels)
3. **Ingest** performance metrics and audience feedback
4. **Iterate** on content strategy based on data

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI SHORTS ENGINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Script  │───▶│  Video   │───▶│  Render  │───▶│ Publish  │          │
│  │   Gen    │    │   Gen    │    │          │    │          │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       ▲                                               │                  │
│       │                                               ▼                  │
│  ┌──────────┐                                   ┌──────────┐            │
│  │ Strategy │◀─────────────────────────────────│ Analytics │            │
│  │  Engine  │                                   │ Ingestion │            │
│  └──────────┘                                   └──────────┘            │
│       ▲                                               │                  │
│       │         ┌──────────┐                         │                  │
│       └─────────│ Comments │◀────────────────────────┘                  │
│                 │ Analysis │                                             │
│                 └──────────┘                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Modularity

The system is built around adapter interfaces, allowing easy swapping of:
- AI video generation providers (OpenAI Sora, Runway, Pika, etc.)
- Rendering engines (FFmpeg, cloud renderers)
- Publishing platforms (YouTube, TikTok, Instagram, etc.)
- Analytics sources (native APIs, third-party tools)

### 2. Asynchronous Processing

All heavy operations run as background jobs via Celery:
- Video generation (can take minutes)
- Rendering and encoding
- Upload to platforms
- Analytics ingestion

### 3. Event-Driven

Jobs communicate through a task queue, enabling:
- Horizontal scaling of workers
- Retry logic with exponential backoff
- Dead letter queues for failed jobs

## Component Details

### Domain Layer (`src/shorts_engine/domain/`)

Core business entities independent of infrastructure:

```python
# Key domain models
Video          # Represents a generated video
VideoRequest   # Request to generate a video
PublishResult  # Result of publishing to a platform
PerformanceMetrics  # Engagement data for a video
```

### Adapter Layer (`src/shorts_engine/adapters/`)

Interfaces and implementations for external services:

#### Video Generation (`adapters/video_gen/`)
- `VideoGenProvider` - Abstract interface for AI video generation
- Implementations: `OpenAISoraProvider`, `RunwayProvider`, `StubProvider`

#### Rendering (`adapters/renderer/`)
- `RendererProvider` - Abstract interface for video rendering
- Implementations: `FFmpegRenderer`, `CloudRenderer`, `StubRenderer`

#### Publishing (`adapters/publisher/`)
- `PublisherAdapter` - Abstract interface for platform publishing
- Implementations: `YouTubeAdapter`, `TikTokAdapter`, `InstagramAdapter`

#### Analytics (`adapters/analytics/`)
- `AnalyticsAdapter` - Abstract interface for fetching metrics
- Implementations: `YouTubeAnalytics`, `TikTokAnalytics`, `StubAnalytics`

#### Comments (`adapters/comments/`)
- `CommentsAdapter` - Abstract interface for fetching comments
- Implementations: `YouTubeComments`, `TikTokComments`, `StubComments`

### Service Layer (`src/shorts_engine/services/`)

Application logic that orchestrates domain and adapters:

- `PipelineService` - Manages end-to-end video generation flow
- `AnalyticsService` - Aggregates metrics across platforms
- `StrategyService` - Determines content strategy based on performance

### Jobs Layer (`src/shorts_engine/jobs/`)

Celery task definitions:

```python
# Task hierarchy
generate_video_task     # Create video from prompt
render_video_task       # Process and encode video
publish_video_task      # Upload to platform
ingest_analytics_task   # Fetch performance metrics
ingest_comments_task    # Fetch and analyze comments
iterate_strategy_task   # Update content strategy
```

### API Layer (`src/shorts_engine/api/`)

FastAPI endpoints for:
- Health checks
- Job triggering
- Status monitoring
- Admin operations

## Data Flow

### Video Generation Pipeline

```
1. API Request / CLI Command / Scheduled Trigger
   │
2. ├── generate_video_task
   │   └── VideoGenProvider.generate(prompt)
   │
3. ├── render_video_task
   │   └── RendererProvider.render(video_data)
   │
4. ├── publish_video_task (per platform)
   │   └── PublisherAdapter.publish(video_file)
   │
5. └── Schedule analytics ingestion
```

### Analytics Ingestion Loop

```
1. Scheduled / API Trigger
   │
2. ├── ingest_analytics_task (per video, per platform)
   │   └── AnalyticsAdapter.fetch_metrics(video_id)
   │
3. ├── ingest_comments_task (per video)
   │   └── CommentsAdapter.fetch_comments(video_id)
   │
4. └── iterate_strategy_task
       └── StrategyService.update(metrics, comments)
```

## Database Schema

```sql
-- Core tables
videos              -- Generated video records
video_requests      -- Generation requests
publish_results     -- Platform publish status
performance_metrics -- Engagement data snapshots
comments           -- Ingested comments
content_strategy   -- Current strategy parameters
```

## Configuration

All configuration via environment variables:

```bash
# Core
DATABASE_URL=postgresql://user:pass@localhost:5432/shorts
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000

# Providers (swap implementations)
VIDEO_GEN_PROVIDER=stub  # openai_sora, runway, pika
RENDERER_PROVIDER=stub   # ffmpeg, cloud
PUBLISHER_YOUTUBE=false
PUBLISHER_TIKTOK=false
PUBLISHER_INSTAGRAM=false
```

## Scaling Considerations

### Horizontal Scaling

- **API**: Stateless, scale with load balancer
- **Workers**: Scale based on queue depth
- **Database**: Read replicas for analytics queries

### Job Priorities

```python
CELERY_TASK_QUEUES = {
    'high': {'priority': 10},    # User-triggered jobs
    'default': {'priority': 5},  # Scheduled jobs
    'low': {'priority': 1},      # Analytics ingestion
}
```

## Future Enhancements

1. **A/B Testing Framework**: Test different content strategies
2. **ML Pipeline Integration**: Train models on performance data
3. **Multi-tenant Support**: Separate channels/accounts
4. **Real-time Dashboard**: Live metrics visualization
5. **Webhook Notifications**: External integrations
