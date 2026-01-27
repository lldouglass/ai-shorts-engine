# AI Shorts Engine

Automated short-form video generation engine. Python 3.11+, FastAPI, Celery, PostgreSQL, Redis.

## Commands
```bash
make up          # Start services (Docker)
make down        # Stop services
make test        # Run tests with coverage
make test-fast   # Tests, stop on first failure
make lint        # ruff + mypy
make format      # black + isort
make migrate     # Run DB migrations
make smoke       # Quick E2E validation
```

## Code Style

- **Line length**: 100 chars (black)
- **Type hints**: Required on all functions
- **Imports**: stdlib → third-party → local, use isort
- **Logging**: Always use structlog, never print()
```python
from shorts_engine.logging import get_logger
logger = get_logger(__name__)
logger.info("event_name", key=value)  # Structured, not f-strings
```

## IMPORTANT: Adapter Pattern

All external services use swappable adapters. NEVER hardcode provider logic.

```
adapters/<type>/base.py    → Abstract base class
adapters/<type>/stub.py    → For testing (no API keys)
adapters/<type>/<impl>.py  → Real implementation
```

When adding a new adapter:
1. Inherit from the base class
2. Implement ALL abstract methods
3. Add provider selection in `__init__.py`
4. Add config setting in `config.py`

## IMPORTANT: Celery Task Patterns

Tasks MUST follow this pattern:
```python
@celery_app.task(bind=True, name="pipeline.task_name")
def task_name(self, job_id: str) -> dict[str, Any]:
    with get_session_context() as session:
        # 1. Idempotency check FIRST (skip if already done)
        # 2. Update status to "running"
        # 3. Do work
        # 4. Return dict with "success" key
```

NEVER use `run_async()` outside of Celery tasks - it's only for calling async code from sync task context.

## IMPORTANT: Domain vs DB Models

- `domain/models.py` - Pure dataclasses, NO SQLAlchemy imports
- `db/models.py` - SQLAlchemy ORM models

Convert at service boundaries. Don't leak DB models into domain logic.

## Testing

- Use stub providers (set `*_PROVIDER=stub` in env)
- Fixtures in `conftest.py` provide stub adapters
- Always verify with `curl http://localhost:8000/health` after `make up`

## Common Mistakes to Avoid

### Database
- NEVER use `session.query()` - use `session.execute(select(...))` (SQLAlchemy 2.0 style)
- ALWAYS use `get_session_context()` context manager, never create sessions manually
- ALWAYS call `session.commit()` after modifications

### Async/Sync
- Celery tasks are sync - use `run_async()` helper to call async adapter methods
- FastAPI routes can be async
- NEVER mix `asyncio.run()` with `run_async()`

### Providers
- Default providers are `stub` - no API keys needed for testing
- Check `config.py` for provider selection logic before adding new ones
- ALWAYS implement stub version first for testing

### Pipelines
- Pipeline stages: plan → generate → verify → render → publish
- Each stage should be idempotent (check if already done before proceeding)
- QA gates run after planning and rendering - don't skip them

### Style Presets
- Defined in `presets/styles.py`
- Add to `PRESETS` dict after defining
- Use `get_preset()` to retrieve, never access dict directly

## Environment Variables

Key settings (see `config.py` for full list):
```bash
LLM_PROVIDER=stub           # openai, anthropic, stub
VIDEO_GEN_PROVIDER=stub     # luma, stub
RENDERER_PROVIDER=stub      # creatomate, stub
QA_ENABLED=true             # Set false to skip QA gates
AUTOPUBLISH_ENABLED=false   # Safety: manual publish by default
```

## Verification

After making changes:
1. `make format` - Auto-fix formatting
2. `make lint` - Check types and style
3. `make test-fast` - Quick test run
4. For pipeline changes: `make smoke` after `make up`
