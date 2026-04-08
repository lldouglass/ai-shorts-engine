# Manifest Bridge Recon

## Goal

Identify the cleanest place in the Python repo to emit a manifest bundle for a separate preset-based TypeScript studio, while keeping Python responsible for generation work:

- script / narrative generation
- reference image generation
- voiceover generation
- lip-sync or scene clip generation
- metadata capture

The TypeScript studio would then consume a manifest bundle instead of asking Python to do the final composition itself.

## Current Pipeline Entry Points

### Stable app pipeline

The DB-backed pipeline under `src/shorts_engine` is the clean integration target.

- CLI entry: `src/shorts_engine/cli.py`
  - `shorts create` enqueues `run_full_pipeline_task`
  - `shorts render` enqueues `run_render_pipeline_task`
- Video-generation orchestrator: `src/shorts_engine/jobs/video_pipeline.py`
  - `run_full_pipeline_task`
  - `plan_job_task`
  - `generate_all_scenes_task`
  - `verify_assets_task`
  - `mark_ready_for_render_task`
- Render orchestrator: `src/shorts_engine/jobs/render_pipeline.py`
  - `generate_voiceover_task`
  - `render_final_video_task`
  - `mark_ready_to_publish_task`
  - optional `critique_final_video_task`

### Script-based pipelines

There are also standalone production scripts that bypass the DB and write directly into `output/`.

- `make_car_short.py`
- `make_listicle_video.py`
- several `scripts/generate_*.py` utilities

These scripts are useful reference implementations for artifact shape, but they are not the cleanest first integration point because they do not normalize outputs into shared models.

## Current Output Directories

### DB-backed pipeline outputs

The shared storage abstraction is `src/shorts_engine/services/storage.py`. By default it writes to `./storage` and creates:

- `storage/clips`
  - scene clips (`asset_type="scene_clip"`)
- `storage/audio`
  - voiceover audio (`asset_type="voiceover"` currently maps to temp unless storage type map is expanded)
- `storage/final`
  - final rendered videos (`asset_type="final_video"`)
- `storage/thumbnails`
- `storage/temp`

Persisted outputs are also represented in `AssetModel` rows in `src/shorts_engine/db/models.py`.

### Script-based outputs

Ad hoc scripts write directly to repo-local `output/` folders:

- `output/car_videos`
  - raw lip-sync MP4s, final MP4s, audio MP3s
- `output/listicle_videos`
  - per-segment audio, raw MP4s, processed MP4s, stitched MP4s, final MP4s
- `output/previews`
  - preview markdown
- `output/samples`
  - sample videos from experimental scripts

## Recommended Integration Point

### Primary recommendation

Plug the future exporter in immediately after `mark_ready_for_render_task` in `src/shorts_engine/jobs/video_pipeline.py`.

Why this is the cleanest seam:

- planning is complete
- scene ordering is fixed
- all scene assets have been verified
- `VideoJobModel`, `SceneModel`, `PromptModel`, `AssetModel`, `StoryModel`, and project metadata are already available
- no render-provider coupling is required
- the exporter can be introduced without changing generation behavior
- it preserves the option to keep the existing Python render pipeline for backward compatibility

At that point the job is effectively "bundle-ready" even if no final MP4 exists yet.

### Suggested flow

Recommended future sequence:

1. `run_full_pipeline_task`
2. `plan_job_task`
3. `generate_all_scenes_from_plan`
4. `verify_assets_task`
5. `mark_ready_for_render_task`
6. new manifest export task

The exporter should either:

- write a manifest bundle and leave the job at `ready`, or
- introduce a new stage such as `manifest_ready` / `bundle_ready`

### When to export later instead

If the TypeScript studio requires voice timing, audio-derived subtitle timings, or final music defaults, a second export mode can run after `generate_voiceover_task` and before `render_final_video_task`.

That mode would have access to:

- narration script
- voiceover asset
- `subtitle_word_boundaries`
- resolved background music settings

This suggests two useful export moments:

- `ready_for_render` export: scene/media bundle only
- `pre_render` export: scene/media bundle plus narration timing

## Data The Exporter Already Has Access To

From the DB-backed pipeline, the exporter can already assemble most of a studio manifest.

### Project/job identity

- `ProjectModel.id`
- `ProjectModel.name`
- `ProjectModel.description`
- `ProjectModel.default_style_preset`
- `ProjectModel.settings`
- `VideoJobModel.id`
- `VideoJobModel.idempotency_key`
- `VideoJobModel.style_preset`
- `VideoJobModel.title`
- `VideoJobModel.description`
- `VideoJobModel.stage`
- `VideoJobModel.created_at`
- `VideoJobModel.metadata_`

### Script/story data

- `VideoJobModel.idea`
- linked `StoryModel` fields when present:
  - `topic`
  - `title`
  - `narrative_text`
  - `narrative_style`
  - `suggested_preset`
  - `word_count`
  - `estimated_duration_seconds`

### Scene structure

From `SceneModel`:

- scene number
- visual prompt
- continuity notes
- caption beat
- duration
- per-scene status
- scene metadata

From `PromptModel`:

- final prompt text
- prompt type
- model used
- version

### Planning metadata

From `VideoJobModel.plan_data`:

- raw planner response
- scene-level planning details not fully broken out into columns

This is the best existing source for preserving planner intent in the manifest.

### Media assets

From `AssetModel`:

- asset type
- scene linkage
- local file path
- URL
- provider
- duration
- width / height when populated
- mime type
- metadata blob

Today this covers at least:

- `scene_clip`
- `voiceover`
- `final_video`

### Render-oriented metadata already available

From `render_pipeline.py`, pre-render or render-time data can include:

- narration script
- subtitle word boundaries
- timed captions derived from TTS timing
- background music URL and volume
- scene order and caption text
- final MP4 URL and asset ID in `job.metadata_`

## Missing Fields For A TypeScript Studio Manifest

The exporter can be added now, but a TS studio manifest will still need additional explicit fields if it is meant to be the source of truth.

### Missing or weakly modeled in the core DB pipeline

- reference images as first-class assets
  - current core pipeline uses frame chaining internally, but does not persist reference images as `AssetModel` rows
- raw lip-sync clips vs final scene clips
  - standalone scripts distinguish raw and processed outputs; the DB-backed pipeline currently only models scene clips
- manifest version/schema version
- bundle root and relative paths
- preset payload expanded for studio use
  - today only the preset name is reliably stored; TS likely needs resolved preset properties too
- shot/motion directives as first-class scene data
  - some of this exists only implicitly in prompts or provider behavior
- transition data between scenes
- character/entity continuity IDs
- speaker metadata per narration segment
- per-scene reference-image lineage
  - source image, previous clip last frame, manual override, etc.
- editing intent metadata
  - hook scene, CTA scene, emphasis moments, lower-thirds, overlays
- caption styling preset
- explicit asset role naming for TS
  - `reference_image`, `lipsync_raw`, `scene_clip_final`, `voiceover_master`, `bgm`, `sfx`, `thumbnail`
- checksums / content hashes for deterministic bundle import
- relative import-safe paths instead of raw local absolute paths

### One concrete storage gap

`StorageService._get_subdir()` does not map `voiceover` explicitly; voiceover bytes currently fall through to `storage/temp` even though the logical asset type is `voiceover`.

That is not a blocker for recon, but it should be normalized before relying on bundles as a durable contract.

## Suggested File And Module Locations

### Exporter module

Recommended path:

- `src/shorts_engine/bridge/export_project_manifest.py`

Reason:

- keeps the bridge separate from providers and from render logic
- makes the manifest exporter reusable from CLI, Celery, and tests
- matches the future role: Python-to-TypeScript handoff

Suggested responsibilities:

- query project/job/story/scenes/assets
- normalize paths and URLs
- resolve preset details from `src/shorts_engine/presets/styles.py`
- emit manifest JSON
- optionally stage a bundle directory with copied or symlinked assets

### Optional service wrapper

If the codebase wants service-style orchestration, add:

- `src/shorts_engine/services/manifest_bridge.py`

Use this for:

- bundle directory planning
- schema versioning
- path normalization
- packaging helpers

Then keep `bridge/export_project_manifest.py` as the thin entry module.

### Celery task wrapper

For minimal disruption, add a small task module rather than embedding export logic into existing tasks:

- `src/shorts_engine/jobs/manifest_bridge.py`

It would expose something like:

- `export_project_manifest_task(video_job_id: str, include_voiceover: bool = False)`

### Output location

Recommended bundle root:

- `storage/manifests/<video_job_id>/`

Recommended contents:

- `manifest.json`
- `assets/` with relative paths or links
- optional `story.txt`
- optional `plan.json`

This keeps manifest bundles near other durable generated artifacts and avoids mixing them with the experimental `output/` tree.

## Recommended Manifest Shape

The exporter should target a bundle centered on `video_job_id`.

Recommended top-level sections:

- `schema_version`
- `project`
- `job`
- `preset`
- `story`
- `scenes`
- `assets`
- `captions`
- `audio`
- `bundle`

Each scene should include:

- stable scene id
- scene number
- duration
- prompt text
- continuity notes
- caption beat
- scene asset refs
- optional timing and transition hints

## Minimal-Disruption Implementation Plan

1. Add exporter module that reads only existing DB models plus preset registry.
2. Add one new Celery task invoked after `mark_ready_for_render_task`.
3. Write manifest to `storage/manifests/<job_id>/manifest.json`.
4. Store manifest path in `VideoJobModel.metadata_`.
5. Leave `run_render_pipeline_task` unchanged.

This keeps the current Python render flow working while enabling a TS studio handoff path.

## Recommendation Summary

Best initial hook:

- after `src/shorts_engine/jobs/video_pipeline.py::mark_ready_for_render_task`

Best initial module:

- `src/shorts_engine/bridge/export_project_manifest.py`

Best task wrapper:

- `src/shorts_engine/jobs/manifest_bridge.py`

Best bundle output:

- `storage/manifests/<video_job_id>/manifest.json`

Main reason:

The DB-backed pipeline already has the right normalized entities and verified scene assets. That gives a stable export seam before Python render logic, with much less disruption than trying to retrofit the standalone `output/` scripts first.
