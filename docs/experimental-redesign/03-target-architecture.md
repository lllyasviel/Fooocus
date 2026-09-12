# Target Architecture

The experimental redesign should evolve the repo from a local Gradio-centered app into an API-first image generation platform.

---

## Current Shape

```text
entry_with_update.py
  -> launch.py
    -> webui.py
      -> modules.async_worker.AsyncTask
      -> modules.async_worker.worker
        -> modules.default_pipeline
          -> modules.core
          -> ldm_patched runtime
```

Current strengths:

- Working generation pipeline.
- Rich feature set.
- Local-first user flow.
- Existing model and LoRA handling.
- Existing metadata and output logging.

Current weaknesses:

- UI, task parsing, worker orchestration, and engine execution are tightly coupled.
- Task data is positional and fragile.
- Runtime state is stored in module-level globals.
- The task queue is in memory.
- Output storage is local-first.
- Observability is print/log based.
- Production deployment is not cleanly separated from local startup.

---

## New Shape

```text
apps/web
  -> browser UI
  -> talks only to API

apps/api
  -> validates requests
  -> creates jobs
  -> stores metadata
  -> exposes job status/events/results

apps/worker
  -> consumes jobs
  -> owns GPU runtime
  -> loads models
  -> runs engine adapter
  -> emits progress events
  -> stores outputs

engine
  -> pure generation domain
  -> no Gradio dependency
  -> no web framework dependency
  -> adapter around existing Fooocus behavior first

packages/schemas
  -> typed shared request/result/event contracts

infra
  -> Docker, compose, deployment, GPU worker config
```

---

## System Diagram

```text
┌─────────────────────────────┐
│          Web App            │
│ Next.js / React Studio UI   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│           API App           │
│ FastAPI, auth, validation   │
└───────┬──────────────┬──────┘
        │              │
        ▼              ▼
┌──────────────┐  ┌──────────────┐
│ Job Database │  │ Asset Store  │
│ Postgres     │  │ Local/S3/R2  │
└──────┬───────┘  └──────────────┘
       │
       ▼
┌─────────────────────────────┐
│        Job Queue            │
│ Redis/NATS/Celery/Dramatiq  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       GPU Worker Pool       │
│ ModelRuntime + EngineAdapter│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Realtime Event Stream   │
│ WebSocket or SSE            │
└─────────────────────────────┘
```

---

## Runtime Design

### ModelRegistry

Responsible for:

- Discovering available checkpoints.
- Discovering LoRAs.
- Discovering VAEs.
- Discovering ControlNet/IP adapter files.
- Recording hash/fingerprint metadata.
- Validating requested model names.

### ModelRuntime

Responsible for:

- Owning loaded model objects.
- Loading base model.
- Loading refiner model.
- Applying LoRAs.
- Loading control models.
- Managing VRAM pressure.
- Reusing warm model state.
- Evicting cold models.

### GenerationPlanner

Responsible for converting a typed generation request into an execution plan.

Example plan stages:

```text
validate_request
resolve_models
resolve_seed
expand_prompt
apply_styles
prepare_inputs
prepare_control_images
prepare_latents
run_sampler
run_refiner
run_decoder
postprocess_outputs
write_metadata
store_assets
emit_result
```

### EngineAdapter

Responsible for calling current Fooocus internals without exposing the old Gradio/task model to the new platform.

First implementation:

```text
FooocusEngineAdapter
  -> accepts GenerationRequest
  -> converts to current pipeline-compatible settings
  -> calls existing engine functions
  -> returns GenerationResult
```

Later implementation:

```text
NativeEngine
  -> decomposed pipeline stages
  -> no old AsyncTask dependency
  -> no module-level global mutation except inside owned runtime context
```

---

## Service Boundaries

| Service | Owns | Must Not Own |
|---|---|---|
| Web App | UI state, forms, canvas, gallery | GPU execution |
| API App | validation, auth, jobs, metadata | direct model objects |
| Worker | GPU runtime, generation execution | public HTTP product logic |
| Engine | image generation domain | user accounts, billing, web UI |
| Storage | asset persistence | generation decisions |
| Model Registry | file discovery, validation, hashes | live request routing |

---

## New Package Layout

```text
apps/
  web/
  api/
  worker/

packages/
  schemas/
  sdk/

engine/
  adapters/
    fooocus_adapter.py
  runtime/
    model_registry.py
    model_runtime.py
    cache_policy.py
  planning/
    generation_planner.py
  pipelines/
    text_to_image.py
    image_to_image.py
    inpaint.py
    upscale.py
    enhance.py
  prompting/
    styles.py
    expansion.py
    wildcards.py
  storage/
    local_store.py
    object_store.py
  events/
    progress.py
  safety/
    image_filter.py

infra/
  docker/
  compose/
  github-actions/

tests/
  unit/
  integration/
  fixtures/
  benchmarks/

docs/
  experimental-redesign/
```

---

## Data Model

Minimum tables/entities:

| Entity | Purpose |
|---|---|
| users | ownership and auth |
| projects | group jobs and assets |
| generation_jobs | job lifecycle |
| generation_inputs | normalized request payloads |
| generation_outputs | final images/assets |
| job_events | progress and audit trail |
| assets | uploads, masks, previews, outputs |
| models | checkpoints and model metadata |
| loras | LoRA metadata |
| presets | reusable settings |
| worker_nodes | worker health and capacity |
| benchmark_runs | performance history |

---

## Job Lifecycle

```text
created
  -> queued
  -> assigned
  -> loading_models
  -> preparing_inputs
  -> running
  -> postprocessing
  -> saving
  -> succeeded
```

Failure/cancel states:

```text
failed
cancel_requested
cancelled
expired
```

---

## Progress Events

Standard event types:

| Event | Purpose |
|---|---|
| job_created | job accepted |
| job_queued | job waiting |
| worker_assigned | worker selected |
| model_loading | model/runtime prep |
| input_preparing | image/mask/preprocessor work |
| sampling_started | diffusion started |
| preview | preview image/status |
| output_saved | output asset stored |
| job_succeeded | job complete |
| job_failed | job failed |
| job_cancelled | job cancelled |

---

## Deployment Modes

| Mode | Purpose |
|---|---|
| Local legacy | Existing app still runs |
| Local experimental | API + worker + local storage |
| Single GPU server | One API, one worker, one GPU |
| Multi-GPU server | API plus multiple worker processes |
| Cloud GPU pool | API separate from GPU hosts |

---

## Migration Strategy

1. Keep current launcher and Gradio path intact.
2. Add new code in parallel directories.
3. Add adapter to call existing engine.
4. Add parity tests.
5. Add API/worker path.
6. Add modern UI.
7. Make Gradio optional.
8. Only remove old paths after parity and benchmarks.
