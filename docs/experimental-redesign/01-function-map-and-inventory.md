# Function Map and Inventory Plan

This document maps the known high-level functions and subsystems in the current repo, then defines how to produce a complete automated function inventory.

The manual map below is not meant to replace automated inventory. It gives architectural meaning to the important paths that must be replicated.

---

## Known Startup Flow

```text
entry_with_update.py
  -> tries to fast-forward update repo through pygit2
  -> imports launch.py

launch.py
  -> sets root path and environment variables
  -> installs torch/torchvision if missing
  -> installs requirements
  -> builds launcher
  -> initializes args/config
  -> downloads required models/assets
  -> initializes hash cache
  -> imports webui.py

webui.py
  -> builds Gradio UI
  -> creates AsyncTask from UI inputs
  -> pushes task into async_worker.async_tasks
  -> streams progress/results back to Gradio

modules/async_worker.py
  -> worker loop consumes AsyncTask objects
  -> preprocesses inputs
  -> applies styles, wildcards, prompt expansion
  -> loads/refreshes models and LoRAs
  -> applies ControlNet/IP adapter/inpaint/upscale/enhance paths
  -> calls modules.default_pipeline.process_diffusion
  -> saves/logs outputs

modules/default_pipeline.py
  -> owns model globals
  -> refreshes base/refiner/LoRA/controlnet state
  -> encodes prompts
  -> runs sampler/refiner/VAE decode

modules/core.py
  -> wraps low-level model loading, VAE encode/decode, ControlNet, FreeU, LoRA patching, latent generation, previewer
```

---

## High-Level Subsystem Map

| Subsystem | Current Files | Responsibility | Redesign Equivalent |
|---|---|---|---|
| Launcher | `entry_with_update.py`, `launch.py` | startup, self-update, install deps, download models | `apps/api`, `apps/worker`, Docker entrypoints |
| CLI/config | `args_manager.py`, `modules/config.py`, presets | command flags, default config, paths | typed config + environment profiles |
| UI | `webui.py`, Gradio extension files | user input, progress, gallery | Next.js/React frontend |
| Task model | `modules/async_worker.AsyncTask` | parse UI args into task fields | `GenerationRequest` schema |
| Worker | `modules/async_worker.worker` | task execution orchestration | GPU worker runtime |
| Pipeline | `modules/default_pipeline.py` | model state, prompt encoding, diffusion process | `InferenceEngine` + `ModelRuntime` |
| Core operations | `modules/core.py` | model load, VAE, sampler wrappers, LoRA, ControlNet | engine adapters/stages |
| Image preprocessing | `extras/preprocessors.py`, inpaint, face crop, SAM, DINO | prepare masks/control images | preprocessing service/stages |
| Prompting/styles | `modules/sdxl_styles.py`, wildcards, expansion | style application and prompt expansion | prompt service |
| Logging/output | `modules/private_logger.py`, metadata parser | save images and generation metadata | storage service + DB metadata |
| Safety | `extras/censor.py` | NSFW blackout/censor behavior | safety service |
| Model management | `modules/model_loader.py`, hash cache, config paths | download/load/cache model files | model registry + cache manager |

---

## Current Function Groups That Must Be Preserved

### Startup and Environment

| Capability | Current Function/Path | Preserve? | Redesign Approach |
|---|---|---:|---|
| Python root setup | `entry_with_update.py`, `launch.py` | Yes | Docker/app entrypoint |
| Optional self-update | `entry_with_update.py` | No for production | Replace with CI/CD deploy |
| Dependency install | `launch.prepare_environment()` | Dev only | Lockfile + Docker image |
| Model download | `launch.download_models()` | Yes | Model registry download/sync command |
| Hash cache | `modules.hash_cache.init_cache` | Yes | Model registry fingerprinting |

### UI and Task Flow

| Capability | Current Function/Path | Preserve? | Redesign Approach |
|---|---|---:|---|
| Create task from UI args | `webui.get_task()` | Behavior yes, implementation no | UI request -> typed `GenerationRequest` |
| Generate click handler | `webui.generate_clicked()` | Behavior yes | API job + realtime progress |
| Stop current task | `stop_clicked()` in `webui.py` | Yes | Job cancellation event |
| Skip current image | `skip_clicked()` in `webui.py` | Yes | Job partial cancel/skip signal |
| Sort enhanced outputs | `webui.sort_enhance_images()` | Yes | Output ordering stage |

### Task Data

| Capability | Current Function/Path | Preserve? | Redesign Approach |
|---|---|---:|---|
| Parse prompt/settings | `AsyncTask.__init__` | Yes | Pydantic schema |
| LoRA list parsing | `get_enabled_loras` call inside `AsyncTask` | Yes | `LoraConfig[]` |
| ControlNet/IP image grouping | `AsyncTask.cn_tasks` | Yes | `ControlImageInput[]` |
| Enhance config parsing | `AsyncTask.enhance_ctrls` | Yes | `EnhanceConfig[]` |
| Metadata options | `save_metadata_to_images`, `metadata_scheme` | Yes | `OutputConfig.metadata` |

### Worker Orchestration

| Capability | Current Function/Path | Preserve? | Redesign Approach |
|---|---|---:|---|
| Worker loop | `modules.async_worker.worker()` | Yes | dedicated worker process |
| Progress updates | `progressbar()` nested function | Yes | `ProgressEvent` stream |
| Intermediate results | `yield_result()` nested function | Yes | result events + asset records |
| Image wall/grid | `build_image_wall()` nested function | Yes | postprocess stage |
| Per-image processing | `process_task()` nested function | Yes | `InferenceEngine.run()` |
| Save/log outputs | `save_and_log()` nested function | Yes | storage + metadata service |
| Apply ControlNet/IP adapters | `apply_control_nets()` nested function | Yes | control stage |
| Variation/upscale prep | `apply_vary()` nested function | Yes | variation/upscale stage |
| Inpaint prep | `apply_inpaint()` nested function | Yes | inpaint stage |

### Pipeline and Model Runtime

| Capability | Current Function/Path | Preserve? | Redesign Approach |
|---|---|---:|---|
| Refresh base model | `refresh_base_model()` | Yes | `ModelRuntime.load_base()` |
| Refresh refiner | `refresh_refiner_model()` | Yes | `ModelRuntime.load_refiner()` |
| Synthetic refiner | `synthesize_refiner_model()` | Yes | refiner strategy |
| Refresh LoRAs | `refresh_loras()` | Yes | LoRA cache/patch stage |
| Prompt encoding cache | `clip_encode_single()` | Yes | bounded text encoder cache |
| Multi-prompt encoding | `clip_encode()` | Yes | text encoder service |
| Clip skip | `set_clip_skip()` | Yes | encoder setting |
| Clear caches | `clear_all_caches()` | Yes | runtime cache controls |
| Prepare text encoder | `prepare_text_encoder()` | Yes | warmup stage |
| Refresh full runtime | `refresh_everything()` | Yes | runtime reconciliation |
| VAE parse/interpose | `vae_parse()` | Yes | VAE strategy |
| Sigma calculation | `calculate_sigmas_all()`, `calculate_sigmas()` | Yes | sampler utility |
| Candidate VAE selection | `get_candidate_vae()` | Yes | VAE strategy |
| Diffusion process | `process_diffusion()` | Yes | core inference stage |

### Core Low-Level Operations

| Capability | Current Function/Path | Preserve? | Redesign Approach |
|---|---|---:|---|
| Model container | `StableDiffusionModel` | Yes | `LoadedModelBundle` |
| Refresh LoRAs on model | `StableDiffusionModel.refresh_loras()` | Yes | LoRA patcher service |
| FreeU patch | `apply_freeu()` | Yes | optional pipeline modifier |
| Load ControlNet | `load_controlnet()` | Yes | model registry/runtime |
| Apply ControlNet | `apply_controlnet()` | Yes | control stage |
| Load checkpoint | `load_model()` | Yes | model loader adapter |
| Empty latent | `generate_empty_latent()` | Yes | latent factory |
| VAE decode | `decode_vae()` | Yes | VAE decoder |
| VAE encode | `encode_vae()` | Yes | VAE encoder |
| VAE inpaint encode | `encode_vae_inpaint()` | Yes | inpaint latent stage |
| Approx previewer | `VAEApprox`, `get_previewer()` | Yes | preview renderer |

---

## Automated Full Function Inventory

A script is added at:

```text
tools/architecture/function_inventory.py
```

It should be run from the repo root:

```bash
python tools/architecture/function_inventory.py --root . --out docs/experimental-redesign/generated-function-inventory.md --json docs/experimental-redesign/generated-function-inventory.json
```

Expected inventory output:

- Every Python module.
- Top-level functions.
- Classes.
- Class methods.
- Async functions.
- Approximate line numbers.
- Import count.
- Module docstring presence.

This is the authoritative map for future refactor tasks.

---

## Function Mapping Acceptance Criteria

The function mapping phase is complete only when:

1. `generated-function-inventory.md` exists.
2. `generated-function-inventory.json` exists.
3. Every Python file is represented or intentionally ignored.
4. Each function group has an assigned redesign home.
5. Each current feature has at least one parity test fixture planned.
6. No current feature is removed without a written replacement or explicit deprecation decision.

---

## Redesign Home Map

| Current Area | New Package |
|---|---|
| `launch.py` startup | `apps/api`, `apps/worker`, `infra/docker` |
| `webui.py` Gradio UI | `apps/web` |
| `modules.async_worker.AsyncTask` | `packages/schemas/generation.py` |
| `modules.async_worker.worker` | `apps/worker/main.py` |
| `modules.default_pipeline` | `engine/runtime`, `engine/pipelines` |
| `modules.core` | `engine/adapters/fooocus_core.py` |
| `modules.private_logger` | `engine/storage`, `apps/api/services/assets.py` |
| `modules.sdxl_styles` | `engine/prompting`, `apps/api/services/prompts.py` |
| `extras.censor` | `engine/safety` |
| `extras.preprocessors` | `engine/preprocessing` |

---

## Notes for Developers

- Start with wrapping current behavior, not rewriting model internals.
- Preserve current output metadata fields wherever possible.
- Keep generated function inventory committed during each major refactor so drift is visible.
- Treat `process_diffusion()` as the core parity boundary until a lower-level engine is safely decomposed.
