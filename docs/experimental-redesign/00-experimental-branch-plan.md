# Experimental Full-System Redesign Plan

Branch: `experimental/full-system-redesign`

Purpose: build a clean, experimental architecture that can replicate current Fooocus functionality while creating a path toward a production-grade AI image studio.

This branch is intentionally experimental. It should not be merged directly into `main` until the current app behavior has been mapped, wrapped, tested, and matched by the new system.

---

## Executive Summary

The existing repo is a working Python/Gradio image generation application with a capable SDXL-style engine, prompt styling, image prompt, inpainting, outpainting, upscale/variation, enhancement, metadata logging, model downloads, and local output handling.

The redesign should preserve the working engine behavior first, then evolve the codebase into a modular platform:

```text
UI -> API -> Job Service -> Queue -> GPU Worker -> Storage -> Realtime Events
```

The safest strategy is not a one-shot rewrite. The correct path is:

1. Map all existing behavior.
2. Define typed request/result schemas.
3. Wrap the existing engine behind an adapter.
4. Add tests that prove parity with current behavior.
5. Build the new API/job/worker architecture around that adapter.
6. Replace Gradio only after the backend contract is stable.
7. Add modern features after parity is proven.

---

## Non-Negotiable Rules

| Rule | Reason |
|---|---|
| Do not break `main` | Current app must remain usable. |
| Every large change gets its own branch | Easier review and rollback. |
| No silent behavior changes | Existing generation behavior must be preserved or intentionally versioned. |
| No positional argument expansion in new code | Use typed schemas. |
| No new global runtime state without an owner object | Needed for testability and multi-worker support. |
| Every feature gets acceptance criteria | Prevents vague rewrites. |
| Every engine change gets benchmarked | Speed claims must be measured. |

---

## Proposed Branch Tree

```text
main
  └─ experimental/full-system-redesign
       ├─ docs/current-function-map
       ├─ docs/replication-matrix
       ├─ feat/generation-schema
       ├─ refactor/engine-adapter
       ├─ test/parity-harness
       ├─ feat/api-job-service
       ├─ feat/queue-worker-runtime
       ├─ feat/storage-abstraction
       ├─ feat/realtime-progress-events
       ├─ feat/nextjs-studio-shell
       ├─ perf/model-runtime-cache
       └─ perf/multi-gpu-scheduler
```

---

## Target Outcome

A redesigned system that can do everything the current app can do, plus:

- API-first generation.
- Durable job queue.
- Replayable generation jobs.
- Multi-user history.
- Structured metadata.
- Real-time progress events.
- Model registry and model cache.
- Multi-GPU worker support.
- Cloud/object storage support.
- Better tests, logs, and benchmarks.
- Modern UI independent from the engine.

---

## Build Order

### Stage 0: Inventory and Docs

Deliverables:

- Function inventory script.
- Manual high-level function map.
- Feature replication matrix.
- Target architecture map.
- Roadmap and task branches.

Exit criteria:

- We can explain what each major subsystem does.
- We know what must be replicated before replacing UI/runtime paths.

### Stage 1: Stabilize Current Behavior

Deliverables:

- Startup smoke test.
- Minimal generation smoke test.
- Baseline benchmark script.
- Current CLI/config snapshot.
- Current output metadata snapshot.

Exit criteria:

- A future branch can prove it did not regress basic generation behavior.

### Stage 2: Typed Generation Contract

Deliverables:

- `GenerationRequest` schema.
- `GenerationResult` schema.
- `ProgressEvent` schema.
- `ModelSelection`, `LoraConfig`, `ControlNetInput`, `InpaintInput`, `EnhanceConfig`, `OutputConfig` schemas.

Exit criteria:

- The old Gradio argument list can be converted into a typed request object.
- The request can be serialized and replayed.

### Stage 3: Engine Adapter

Deliverables:

- `FooocusEngineAdapter` wrapping current pipeline calls.
- No UI dependency inside adapter.
- No Gradio-specific inputs inside adapter.
- Adapter returns typed result objects.

Exit criteria:

- A Python script can call generation without launching Gradio.

### Stage 4: Job Service and Queue

Deliverables:

- FastAPI endpoint to create jobs.
- Durable job records.
- Redis/NATS/Celery/Dramatiq worker path.
- Cancel/retry/status events.

Exit criteria:

- A job can be submitted through API and executed by a separate worker process.

### Stage 5: Modern Studio UI

Deliverables:

- Next.js/React studio shell.
- Prompt panel.
- Settings panel.
- Realtime progress.
- Gallery/history.
- Replay button.

Exit criteria:

- Gradio is no longer required for normal user interaction.

### Stage 6: Performance and Scale

Deliverables:

- Model warm cache.
- VRAM-aware worker scheduler.
- Preview throttling.
- Multi-GPU routing.
- Benchmarks dashboard.

Exit criteria:

- Speed and memory improvements are measured, not guessed.

---

## Current Repo Risk Areas

| Area | Risk | Redesign Fix |
|---|---|---|
| Gradio UI owns workflow | Hard to scale or productize | API-first frontend/backend split |
| Positional task args | Fragile and hard to test | Typed request schema |
| Global model state | Hard to run multiple workers safely | Runtime context/model registry |
| In-memory task list | Jobs disappear on restart | Durable queue + database |
| Local output handling | Not cloud-ready | Storage abstraction |
| Print logging | Hard to debug production | Structured logs + metrics |
| Self-update launcher | Dangerous for production | Versioned deploy pipeline |

---

## Definition of Done for Experimental System

The experimental system is considered successful when:

1. It can generate an image from text with the same base settings as current Fooocus.
2. It can run image prompt/control image workflows.
3. It can run upscale/variation.
4. It can run inpainting/outpainting.
5. It can run enhance workflows.
6. It can save metadata and replay jobs.
7. It can stream progress events.
8. It can store outputs outside the local Gradio temp path.
9. It has parity tests against known request fixtures.
10. It has performance benchmarks showing cold start, warm start, steps/sec, VRAM, and output time.

---

## Do Not Build Yet

Avoid these until parity is proven:

- Paid billing.
- Marketplace.
- Team collaboration.
- Mobile app.
- Arbitrary plugin marketplace.
- Replacing the sampler math wholesale.
- Removing Gradio before the API/worker path is stable.

---

## First Engineering Tasks

1. Run the function inventory tool added in this branch.
2. Review generated function inventory.
3. Add smoke tests for current launch and generation path.
4. Create typed generation schema.
5. Build adapter around the current `modules.default_pipeline` and `modules.async_worker` behavior.
6. Add replayable JSON fixtures for text-to-image, image prompt, upscale, inpaint, enhance.
