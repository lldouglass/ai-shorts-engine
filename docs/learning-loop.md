# Learning Loop System

The Learning Loop is an automated system for continuously improving video performance by learning from past results and systematically testing variations.

## Overview

### How It Works

1. **Recipe Tracking**: Each video is associated with a "recipe" - a combination of production parameters:
   - `preset`: Visual style (DARK_DYSTOPIAN_ANIME, VIBRANT_MOTION_GRAPHICS, etc.)
   - `hook_type`: Opening style (question, statement, visual, story, contrast, mystery)
   - `scene_count`: Number of scenes (5-10)
   - `narration_wpm_bucket`: Narration speed (slow, medium, fast)
   - `caption_density_bucket`: On-screen text frequency (sparse, medium, dense)
   - `ending_type`: How the video ends (cliffhanger, resolve, cta, loop)

2. **Performance Measurement**: Videos are scored using a weighted reward formula:
   ```
   reward = 0.5 * avg_view_duration_score + 0.3 * views_6h_score + 0.2 * engagement_score
   ```
   All components are normalized using percentile ranking within the project.

3. **Exploit/Explore Strategy**:
   - **70% Exploit**: New videos use top-performing recipes
   - **30% Explore**: New videos test variations (A/B tests)

4. **Experiment Tracking**: Explore videos are tracked as experiments, comparing the mutated recipe against the baseline.

## Reward Score Calculation

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Average View Duration | 50% | How long viewers watch |
| Views in First 6 Hours | 30% | Initial viral potential |
| Engagement Rate | 20% | (likes + comments) / views |

### Normalization

All metrics are converted to percentile scores (0-1) using the project's historical data from the last 30 days. This ensures fair comparison across different content types and audience sizes.

## Recipe Definition

A recipe captures the key parameters that influence video performance:

```python
Recipe(
    preset="DARK_DYSTOPIAN_ANIME",
    hook_type="question",
    scene_count=7,
    narration_wpm_bucket="medium",
    caption_density_bucket="medium",
    ending_type="cliffhanger",
)
```

### Hook Types

| Type | Description |
|------|-------------|
| `question` | Opens with a compelling question |
| `statement` | Bold, attention-grabbing statement |
| `visual` | Eye-catching visual moment |
| `story` | Mini-story opening |
| `contrast` | Before/after or comparison |
| `mystery` | Creates curiosity gap |

### Ending Types

| Type | Description |
|------|-------------|
| `cliffhanger` | Leaves viewers wanting more |
| `resolve` | Complete conclusion |
| `cta` | Call to action |
| `loop` | Loops back to beginning |

### Buckets

**Narration WPM:**
- `slow`: < 120 words per minute
- `medium`: 120-160 WPM
- `fast`: > 160 WPM

**Caption Density:**
- `sparse`: < 0.3 captions per second
- `medium`: 0.3-0.6 captions per second
- `dense`: > 0.6 captions per second

## Safeguards

### Duplicate Prevention

The system prevents duplicate recipe+topic combinations using a hash. If you try to create a video with the same topic and recipe as an existing video, the sampler will:
1. Try alternative recipes
2. If all recipes are exhausted, skip that topic

### Explore Constraints

- Scene count is capped at 5-10 scenes
- Only one variable is mutated at a time for clear A/B testing
- Experiments track baseline vs variant performance

## API Endpoints

### GET /api/v1/admin/recommendations

Get recipe recommendations for the next batch.

**Query Parameters:**
- `project_id` (required): Project UUID
- `n`: Number of recommendations (default: 5)
- `topics`: Comma-separated topic list

**Response:**
```json
{
  "recommendations": [
    {
      "type": "exploit",
      "recipe": {...},
      "topic": "...",
      "reason": "Based on top-performing recipe"
    }
  ],
  "top_recipes": [...],
  "running_experiments": 3,
  "recent_batches": 2
}
```

### GET /api/v1/admin/recipes

List all recipes for a project.

### GET /api/v1/admin/experiments

List A/B test experiments.

### GET /api/v1/admin/stats

Get aggregate learning loop statistics.

### POST /api/v1/admin/backfill-recipes

Extract recipes from existing videos.

### POST /api/v1/admin/update-stats

Trigger stats recalculation.

## CLI Commands

### Plan Next Batch

```bash
shorts-engine learn plan-next \
  --project <uuid> \
  --topics-file topics.txt \
  --count 5 \
  --wait
```

### View Recipes

```bash
shorts-engine learn recipes --project <uuid>
```

### View Experiments

```bash
shorts-engine learn experiments --project <uuid> --status running
```

### View Stats

```bash
shorts-engine learn stats --project <uuid>
```

### Backfill Recipes

```bash
shorts-engine learn backfill --project <uuid>
```

### Update Stats

```bash
shorts-engine learn update-stats --project <uuid> --days 30
```

## Celery Tasks

### plan_next_batch

Plans and creates the next batch of video jobs.

```python
from shorts_engine.jobs.learning_jobs import plan_next_batch_task

result = plan_next_batch_task.delay(
    project_id="<uuid>",
    n=5,
    topics=["Topic 1", "Topic 2", ...],
)
```

### update_recipe_stats

Updates recipe statistics (runs daily via beat scheduler).

### evaluate_experiments

Evaluates running experiments and marks completed ones (runs daily).

## Database Schema

### recipes

Stores canonical recipe definitions:
- `id`, `project_id`
- Recipe components: `preset`, `hook_type`, `scene_count`, etc.
- `recipe_hash`: Unique identifier for deduplication
- Performance: `times_used`, `avg_reward_score`, `best_reward_score`

### experiments

Tracks A/B test experiments:
- `variable_tested`: Which recipe component is being tested
- `baseline_value`, `variant_value`: The values being compared
- `baseline_avg_reward`, `variant_avg_reward`: Results
- `winner`: `baseline`, `variant`, or `inconclusive`

### planned_batches

Records nightly batch planning:
- `batch_date`
- `exploit_count`, `explore_count`
- `jobs_created`, `jobs_completed`

### video_jobs (updated)

New columns:
- `recipe_id`: Link to recipe
- `experiment_id`: Link to experiment (if explore mode)
- `generation_mode`: `exploit`, `explore`, or `manual`
- `batch_id`: Link to batch
- `topic_hash`: For deduplication

## Integration with Video Pipeline

When planning a video with a recipe, use `LearningLoopPlanner`:

```python
from shorts_engine.services.learning import LearningLoopPlanner, Recipe

planner = LearningLoopPlanner()
recipe = Recipe(
    preset="DARK_DYSTOPIAN_ANIME",
    hook_type="question",
    scene_count=7,
    narration_wpm_bucket="medium",
    caption_density_bucket="medium",
    ending_type="cliffhanger",
)

plan = await planner.plan(idea="A lone samurai...", recipe=recipe)
```

The planner enforces:
- Exact scene count from recipe
- Hook type guidance for Scene 1
- Ending type guidance for final scene
- Narration pacing constraints
