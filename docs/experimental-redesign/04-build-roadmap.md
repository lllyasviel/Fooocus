# Experimental Build Roadmap

This roadmap turns the redesign into small, reviewable tasks. Each task should be developed on its own branch from `experimental/full-system-redesign` unless it is a tiny docs-only update.

---

## Phase 0: Inventory and Safety Rails

Goal: understand current repo behavior and prevent accidental regressions.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Generate function inventory | `docs/current-function-inventory` | generated inventory markdown/json | All Python files scanned |
| Add feature parity checklist | `docs/feature-parity-checklist` | parity checklist | Every user-facing feature assigned status |
| Add startup smoke test | `test/startup-smoke` | test script | App imports/starts without launching full UI where possible |
| Add dependency audit | `chore/dependency-audit` | dependency notes | Known outdated/high-risk dependencies listed |
| Add benchmark baseline | `perf/baseline-benchmark` | timing script | Cold start and warm generation metrics captured |

---

## Phase 1: Typed Contracts

Goal: stop relying on positional UI arguments and create replayable generation jobs.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add schema package | `feat/schema-package` | `packages/schemas` | Importable without heavy GPU deps |
| Add generation request schema | `feat/generation-request-schema` | `GenerationRequest` | Covers prompt, seed, models, LoRAs, size, advanced settings |
| Add image input schemas | `feat/image-input-schemas` | control/inpaint/upscale/enhance inputs | Covers current image workflows |
| Add result schema | `feat/generation-result-schema` | `GenerationResult` | Outputs, metadata, timing, errors |
| Add progress event schema | `feat/progress-event-schema` | `ProgressEvent` | Supports all known progress states |
| Add schema tests | `test/schema-validation` | pytest tests | Valid and invalid fixtures covered |

---

## Phase 2: Engine Adapter

Goal: call existing generation behavior without Gradio.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Create engine package | `refactor/create-engine-package` | `engine/` scaffold | No behavior change |
| Add Fooocus adapter interface | `refactor/fooocus-engine-adapter` | adapter class | Accepts typed request |
| Add Gradio arg translator | `refactor/gradio-arg-translator` | converter utility | Current task settings can become typed request |
| Add dry-run planner | `feat/generation-dry-run-planner` | stage plan output | Can inspect pipeline steps without GPU |
| Add direct generation CLI | `feat/direct-generation-cli` | CLI script | Can submit one request outside Gradio |
| Add adapter tests | `test/engine-adapter` | tests | Mocked generation path passes |

---

## Phase 3: Job System

Goal: move from in-memory task list to durable generation jobs.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add API app scaffold | `feat/api-app-scaffold` | `apps/api` | Starts with health endpoint |
| Add job models | `feat/job-models` | DB models/migrations | Job lifecycle stored |
| Add generation create endpoint | `feat/generation-create-endpoint` | HTTP endpoint | Validates request and creates job |
| Add queue adapter | `feat/queue-adapter` | Redis/NATS abstraction | Can enqueue/dequeue job IDs |
| Add worker app scaffold | `feat/worker-app-scaffold` | `apps/worker` | Worker can poll/consume queue |
| Add cancellation flow | `feat/job-cancellation` | cancel endpoint + worker check | Running jobs can be cancelled safely |
| Add retries/failure capture | `feat/job-retries-errors` | error model | Failures are stored and visible |

---

## Phase 4: Storage and Metadata

Goal: outputs and uploads become platform assets instead of local-only files.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add asset model | `feat/asset-model` | asset DB records | Uploads/outputs are addressable |
| Add local storage adapter | `feat/local-storage-adapter` | local file storage | Works in dev |
| Add object storage adapter | `feat/object-storage-adapter` | S3/R2-compatible adapter | Configurable in env |
| Add metadata writer | `feat/metadata-writer` | metadata service | Existing metadata keys preserved |
| Add output thumbnails | `feat/output-thumbnails` | thumbnail pipeline | Gallery can load quickly |
| Add replay from metadata | `feat/replay-from-metadata` | replay utility | Saved job can regenerate request |

---

## Phase 5: Realtime Progress

Goal: replace Gradio polling with proper realtime job events.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add job event table | `feat/job-event-table` | persisted events | Events queryable per job |
| Add event emitter | `feat/job-event-emitter` | worker emits events | Runtime stages visible |
| Add SSE endpoint | `feat/sse-job-events` | event stream | Browser can subscribe |
| Add preview asset events | `feat/preview-events` | preview assets | Preview updates are throttled |
| Add event replay | `feat/event-replay` | event history endpoint | Reloaded UI can reconstruct progress |

---

## Phase 6: Modern UI

Goal: build a studio UI that talks to the API only.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add web app scaffold | `feat/web-studio-shell` | `apps/web` | Runs independently |
| Add prompt/settings form | `feat/web-generation-form` | UI form | Creates typed request |
| Add live progress view | `feat/web-live-progress` | realtime progress | Consumes SSE/WebSocket events |
| Add output gallery | `feat/web-output-gallery` | gallery/history | Shows stored outputs |
| Add replay button | `feat/web-replay-job` | replay UI | Recreates request from history |
| Add image input UI | `feat/web-image-inputs` | upload/mask/control image UI | Supports current image workflows |

---

## Phase 7: Performance and Scaling

Goal: make the new architecture faster and measurable.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add model warm cache | `perf/model-warm-cache` | runtime cache | Repeated jobs avoid reload where safe |
| Add VRAM-aware scheduling | `perf/vram-aware-scheduler` | worker capacity model | Jobs route by requirements |
| Add preview throttling | `perf/preview-throttling` | event rate control | Preview does not slow generation |
| Add multi-worker support | `perf/multi-worker-runtime` | multiple workers | Jobs distribute correctly |
| Add benchmark dashboard | `perf/benchmark-dashboard` | benchmark outputs | Performance visible over time |

---

## Phase 8: Product Platform Features

Goal: move from tool to product.

| Task | Branch | Output | Acceptance Criteria |
|---|---|---|---|
| Add auth | `feat/auth` | users/sessions | User owns jobs/assets |
| Add projects | `feat/projects` | project folders | Jobs can be grouped |
| Add presets | `feat/presets` | reusable settings | Save/load generation configs |
| Add API keys | `feat/api-keys` | developer API auth | External clients can generate |
| Add admin console | `feat/admin-console` | model/job/worker admin | Operators can manage system |
| Add cost tracking | `feat/cost-tracking` | runtime/cost model | Generation cost visible |

---

## Review Gates

No phase should move forward without these gates:

| Gate | Requirement |
|---|---|
| Tests | New behavior has tests or documented manual validation |
| Docs | Any new subsystem has an architecture note |
| Rollback | Branch can be reverted without breaking unrelated work |
| Performance | Engine changes include timing before/after where relevant |
| Parity | Current feature behavior is preserved or explicitly changed |

---

## First Code Branch Recommendation

After this docs branch, the first code branch should be:

```text
feat/generation-request-schema
```

Why first:

- It does not require GPU.
- It reduces risk.
- It creates the contract for API, worker, tests, and UI.
- It starts replacing the fragile positional task model.

Minimum files for that branch:

```text
packages/schemas/__init__.py
packages/schemas/generation.py
packages/schemas/events.py
packages/schemas/assets.py
tests/schemas/test_generation_request.py
tests/fixtures/generation/text_to_image_basic.json
```

---

## Second Code Branch Recommendation

```text
refactor/fooocus-engine-adapter
```

Goal:

- Create a wrapper around existing generation behavior.
- Do not rewrite sampler/model internals yet.
- Make generation callable without Gradio.

Minimum files:

```text
engine/adapters/fooocus_adapter.py
engine/planning/generation_planner.py
engine/results.py
tools/run_generation_request.py
tests/engine/test_generation_planner.py
```

---

## Third Code Branch Recommendation

```text
feat/api-job-service
```

Goal:

- Add API scaffolding.
- Accept typed generation requests.
- Store jobs.
- Return job IDs.

Minimum files:

```text
apps/api/main.py
apps/api/routes/generation.py
apps/api/services/jobs.py
apps/api/models/jobs.py
```

---

## Experimental Branch Success Definition

The experimental branch has succeeded when a developer can:

1. Start an API service.
2. Start a GPU worker.
3. Submit a generation request through HTTP.
4. Watch progress events.
5. Retrieve output assets.
6. Replay a job from saved metadata.
7. Compare runtime to baseline.
8. Keep the existing Gradio path available during migration.
