# Learning Loop Runbook

This runbook describes how to operate the learning loop for daily video generation.

## Prerequisites

1. **Database Migration**: Run migration 0005 to create learning loop tables:
   ```bash
   alembic upgrade head
   ```

2. **Active Project**: Have a project with published videos and metrics data.

3. **Celery Workers**: Ensure workers are running with the `learning` queue:
   ```bash
   celery -A shorts_engine.worker worker -Q learning,high,low,default --loglevel=info
   ```

4. **Beat Scheduler**: For automated daily tasks:
   ```bash
   celery -A shorts_engine.worker beat --loglevel=info
   ```

## Daily Operations

### Step 1: Prepare Topics

Create a file with video topic ideas (one per line):

```bash
# topics.txt
A mysterious warrior walks through an abandoned temple
The last sunset before the apocalypse
A secret meeting in a neon-lit alley
An ancient artifact awakens after centuries
Two rivals face off in a destroyed cityscape
```

### Step 2: Plan the Batch

Run the batch planning command:

```bash
shorts-engine learn plan-next \
  --project YOUR_PROJECT_UUID \
  --topics-file topics.txt \
  --count 5 \
  --wait
```

This will:
- Sample 70% exploit recipes (top performers)
- Sample 30% explore recipes (A/B test mutations)
- Create video jobs with recipe assignments
- Record experiments for tracking

### Step 3: Execute Video Jobs

For each created job, run the full pipeline:

```bash
# List the jobs from the batch
shorts-engine shorts list --project YOUR_PROJECT_UUID --limit 10

# Run each job (or automate this)
shorts-engine shorts create --project YOUR_PROJECT_UUID --idea "..." --wait
```

Alternatively, programmatically trigger pipelines for batch jobs.

### Step 4: Render and Publish

```bash
# Render
shorts-engine shorts render --job JOB_UUID --wait

# Publish
shorts-engine shorts publish --job JOB_UUID --youtube-account "Main Channel" --wait
```

### Step 5: Wait for Metrics

Metrics are automatically ingested hourly via the Celery beat schedule. You can also trigger manually:

```bash
shorts-engine ingest metrics --since 24h --wait
```

### Step 6: Update Recipe Stats

After metrics are collected (wait at least 6-24 hours), update recipe statistics:

```bash
shorts-engine learn update-stats --project YOUR_PROJECT_UUID --days 30
```

### Step 7: Review Results

```bash
# View top recipes
shorts-engine learn recipes --project YOUR_PROJECT_UUID

# View experiment results
shorts-engine learn experiments --project YOUR_PROJECT_UUID

# View overall stats
shorts-engine learn stats --project YOUR_PROJECT_UUID
```

## Automated Cron Schedule

For fully automated operation, set up a cron job:

```bash
# /etc/cron.d/shorts-engine-learning

# Plan batch at 6 AM UTC daily
0 6 * * * cd /app && shorts-engine learn plan-next --project UUID --topics-file /data/topics/$(date +\%Y-\%m-\%d).txt --count 5

# Update stats at 2 AM UTC daily (after metrics collected)
0 2 * * * cd /app && shorts-engine learn update-stats --project UUID --days 30
```

## Topic Generation

Topics can be generated from various sources:

### Manual Curation
Create topics based on trending themes or content strategy.

### LLM Generation
```python
# Example: Generate topics using LLM
topics = await llm.complete([
    LLMMessage(role="system", content="Generate 10 short video topic ideas in the dark anime style."),
])
```

### Trend Scraping
Scrape trending topics from social media, news, or keyword tools.

## Monitoring

### Health Checks

```bash
# Check if API is healthy
curl http://localhost:8000/health/ready

# Check Celery workers
celery -A shorts_engine.worker inspect active
```

### Key Metrics to Monitor

1. **Batch Success Rate**: % of planned batches that complete
2. **Exploit vs Explore Ratio**: Should be ~70/30
3. **Experiment Completion**: Running experiments should resolve within 2-4 weeks
4. **Average Reward Trend**: Should improve over time

### Alerts

Set up alerts for:
- Batch planning failures
- Zero videos created in a batch
- Experiment backlog growing too large
- Reward score decline

## Troubleshooting

### No Recipes Found

If `plan-next` fails with "No top recipes":

1. Check if videos have recipe features:
   ```bash
   shorts-engine learn stats --project UUID
   ```

2. Backfill recipes from existing videos:
   ```bash
   shorts-engine learn backfill --project UUID
   ```

### Duplicate Topic Errors

If many topics are being skipped as duplicates:

1. Use more varied topics
2. Check existing topic hashes in database
3. The system will automatically try alternative recipes

### Low Reward Scores

If scores are consistently low:

1. Review the top-performing recipes
2. Check if metrics are being ingested properly
3. Consider manual recipe creation with different parameters

### Experiment Not Completing

Experiments need at least 5 videos per variant (baseline + variant). Check:

1. Are explore videos being created?
2. Are they being published?
3. Are metrics being collected?

## Best Practices

### Topic Quality

- Keep topics concise but descriptive
- Avoid duplicating recent topics
- Match topics to your content niche

### Batch Size

- Start with 3-5 videos per day
- Scale based on production capacity
- Consider platform upload limits

### Recipe Diversity

- Allow the system to explore for at least 2 weeks
- Don't manually override recipes too often
- Trust the data over intuition

### Metric Collection Timing

- Wait at least 6 hours before analyzing 6h metrics
- Wait 24 hours before making recipe decisions
- Weekly review for recipe performance trends

## Example Full Workflow

```bash
# Day 1: Setup
alembic upgrade head
shorts-engine learn backfill --project $PROJECT_ID

# Day 2+: Daily batch
echo "Epic battle scene
Hidden temple discovery
Last survivor's journey
Cyberpunk chase sequence
Ancient prophecy unfolds" > /tmp/topics.txt

shorts-engine learn plan-next \
  --project $PROJECT_ID \
  --topics-file /tmp/topics.txt \
  --count 5 \
  --wait

# Get job IDs from output, then for each:
shorts-engine shorts create --project $PROJECT_ID --idea "..." --preset DARK_DYSTOPIAN_ANIME --wait
shorts-engine shorts render --job $JOB_ID --wait
shorts-engine shorts publish --job $JOB_ID --youtube-account "Main" --wait

# Next day: Check results
shorts-engine learn stats --project $PROJECT_ID
shorts-engine learn recipes --project $PROJECT_ID
shorts-engine learn experiments --project $PROJECT_ID
```
