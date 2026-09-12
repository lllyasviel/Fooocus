# Fooocus Image Command Center Design

## Goal

Turn Fooocus from a single-user Gradio generation app into a controlled image-generation platform.

Fooocus should remain the generation engine. A new control plane owns jobs, queues, gallery metadata, prompt tracing, compute routing, RunPod provisioning policy, and future UI control.

## Current Problem

The existing app is fast to use locally, but it is hard to scale because generation is tied to one running UI process and one local worker flow. The system needs a source of truth outside the Gradio screen.

The target design avoids breaking the current Fooocus app. It adds a new backend layer first, then the UI can be upgraded after the backend is stable.

## Target Architecture

```txt
React UI or Gradio bridge
        |
        v
FastAPI Control Plane
        |
        +-- SQLite now / Postgres later
        +-- Persistent output/gallery folder
        +-- Queue manager
        +-- Compute router
        |       +-- local worker
        |       +-- RunPod worker
        |       +-- safe RunPod provisioner
        +-- Trace/prompt builder service
        +-- Gallery service
```

## Backend Source of Truth

The control plane should own:

- job IDs
- job status
- queue placement
- compute target
- prompt metadata
- source image metadata
- generated image metadata
- gallery metadata
- cost metadata
- worker health

Fooocus/Gradio should become one client of the control plane, not the only source of truth.

## Job Types

- `trace`: analyze an uploaded image and create a smarter prompt.
- `generate`: normal text-to-image or image-to-image.
- `upscale`: upscale, repair, variation, enhancement.
- `batch`: multi-prompt/multi-variation work.
- `admin`: health check, model warmup, cleanup.

## Queues

```txt
trace_queue      image understanding and prompt building
fast_queue       light image generations
heavy_queue      SDXL, upscale, high-res, refiner, inpaint
priority_queue   admin/manual priority work
batch_queue      bulk background work
```

## Compute Switch

The system supports three modes:

- `local`: only use local GPU/worker.
- `runpod`: only use RunPod worker.
- `auto`: use local if healthy and available; otherwise use RunPod.

Hot switch rules:

- Running jobs finish where they started.
- New jobs use the latest compute mode.
- Each job records where it ran.
- A failed RunPod route may fall back to local if local is healthy.

## RunPod Auto Provisioning

Provisioning must be safe by default. The app should never spend money silently.

Default policy:

```txt
RUNPOD_AUTO_PROVISION=false
RUNPOD_REQUIRE_APPROVAL=true
RUNPOD_MAX_HOURLY_COST=1.00
RUNPOD_MAX_SESSION_COST=5.00
RUNPOD_IDLE_SHUTDOWN_MINUTES=15
RUNPOD_ALLOWED_GPUS=RTX_4090,L40S,A40,A100
```

Provisioning flow:

```txt
Job needs RunPod
  -> no healthy RunPod worker
  -> check policy
  -> request user approval if required
  -> create pod only if approved and inside budget
  -> wait for health endpoint
  -> route job
  -> stop pod after idle timeout
```

## Gallery

Every output should create a gallery row with:

- image id
- job id
- output path
- thumbnail path
- prompt
- negative prompt
- seed
- model
- LoRAs
- source image path
- trace id
- dimensions
- compute target
- worker id
- generation time
- estimated cost
- favorite flag
- timestamps

Gallery actions:

- reuse prompt
- generate similar
- trace image
- upscale
- favorite
- delete/restore
- export

## Auto Trace / Image Understanding

Trace should convert a reference image into structured prompt intelligence:

```json
{
  "caption": "short natural description",
  "subjects": ["main objects"],
  "style_tags": ["cinematic", "realistic", "product photo"],
  "colors": ["dominant colors"],
  "lighting": "lighting description",
  "composition": "camera angle/framing/layout",
  "quality_tags": ["high detail", "sharp"],
  "suggested_prompt": "clean editable prompt",
  "suggested_negative_prompt": "clean negatives"
}
```

First use existing Fooocus/BLIP-style interrogation if available. Later adapters can use Florence-2, Moondream, or LLaVA behind the same interface.

## API Contract

```txt
GET  /health
GET  /compute/status
POST /compute/mode
POST /compute/test-local
POST /compute/test-runpod
POST /jobs
GET  /jobs/{id}
POST /jobs/{id}/cancel
POST /jobs/{id}/retry
GET  /queues
POST /trace-image
GET  /gallery
GET  /gallery/{id}
POST /gallery/{id}/favorite
POST /gallery/{id}/reuse
DELETE /gallery/{id}
```

## Build Order

1. Add non-breaking control-plane package.
2. Add SQLite-backed job/gallery store.
3. Add queue selector.
4. Add compute router.
5. Add safe RunPod policy manager.
6. Add trace service contract.
7. Add optional FastAPI endpoints.
8. Wire completed generation outputs into gallery metadata.
9. Add minimal Gradio buttons.
10. Build full React UI after backend stabilizes.

## Acceptance Criteria

- Existing Fooocus still runs.
- `main` remains untouched.
- Jobs can be created with IDs and persisted status.
- Gallery metadata persists after restart.
- Compute mode can switch without restarting.
- RunPod cannot spend money without explicit approval by default.
- Trace jobs do not block generation jobs.
- Outputs can be reused from gallery metadata.
