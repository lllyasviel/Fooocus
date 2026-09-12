# Fooocus — MiamiCreme Engine Modernization Roadmap

This repository is a MiamiCreme fork of Fooocus used to plan and validate a safer, faster, easier-to-use image generation engine while preserving the working Stable Diffusion XL/Fooocus core.

The original Fooocus project is an offline, open-source image generation application built around Gradio and SDXL. This fork keeps that foundation, but adds a structured roadmap for performance, usability, maintainability, architecture documentation, and future redesign work.

> Important: this fork should protect the existing engine first. Major engine changes should be developed on separate branches, validated through repeatable generation tests, and merged only after the stable paths still work.

## Branch Map

| Branch | Status | Purpose | What It Contains | Merge Guidance |
| --- | --- | --- | --- | --- |
| `main` | Stable baseline | Current safe branch for the fork | Existing Fooocus codebase plus this README roadmap | Keep stable. Do not use as a scratch branch. Only merge tested, reviewed updates. |
| [`experimental/full-system-redesign`](https://github.com/miamicreme/Fooocus/tree/experimental/full-system-redesign) | Existing experimental branch | Full-system redesign planning branch | Adds architecture and redesign documentation under `docs/experimental-redesign/` plus `tools/architecture/function_inventory.py` | Planning and discovery branch only. Do not merge blindly into `main`; use it to guide future implementation branches. |
| `develop` | Recommended future branch | Integration branch for approved work | Should become the staging area for tested Phase 1 and Phase 2 improvements | Merge feature branches here first. Validate before promoting to `main`. |
| `perf/phase-1-fast-path` | Recommended future branch | Safe speed improvements | Preview frequency control, smarter CLIP cache behavior, fast presets, refiner-off defaults for speed modes, draft-mode save/metadata options | Low-risk branch. Keep changes reversible and flag-driven. |
| `engine/phase-2-core-hardening` | Recommended future branch | Engine structure and queue cleanup | Split refresh paths, safer model refresh logic, thread-safe job queue, preprocessing cache, async warmup fallback | Medium-risk branch. Must pass full generation-path regression checks before merge. |
| `ui/preset-driven-workflow` | Recommended future branch | Easier user experience | Beginner/Pro mode, Draft/Fast/Balanced/Final presets, better labels, slow-feature warnings | Safe if it does not change sampler behavior directly. |
| `qa/regression-matrix` | Recommended future branch | Validation and testing | Smoke tests, run checklist, generation-path validation notes, reproducible manual test matrix | Should be created before deep engine refactors. |
| `docs/architecture-roadmap` | Recommended future branch | Architecture documentation | Engine maps, dependency maps, pipeline diagrams, branch strategy, implementation notes | Documentation-only; safe to merge once accurate. |

## Current Experimental Redesign Branch

The branch [`experimental/full-system-redesign`](https://github.com/miamicreme/Fooocus/tree/experimental/full-system-redesign) is currently ahead of `main` and contains planning assets rather than direct engine rewrites.

Files added by that branch:

| File | Purpose |
| --- | --- |
| `docs/experimental-redesign/README.md` | Entry point for the redesign documentation set. |
| `docs/experimental-redesign/00-experimental-branch-plan.md` | Explains how the experimental branch should be used and protected. |
| `docs/experimental-redesign/01-function-map-and-inventory.md` | Maps existing functions and identifies areas that need deeper review. |
| `docs/experimental-redesign/02-feature-replication-checklist.md` | Tracks original Fooocus feature coverage so a redesign does not lose behavior. |
| `docs/experimental-redesign/03-target-architecture.md` | Describes the future target architecture and separation of concerns. |
| `docs/experimental-redesign/04-build-roadmap.md` | Lays out the staged build plan for the redesign. |
| `tools/architecture/function_inventory.py` | Utility script for scanning and inventorying repository functions. |

This branch is valuable because it documents the system before rewriting it. It should be treated as the architecture discovery branch, not as a production engine branch.

## Future Update Strategy

The modernization plan should happen in phases. The goal is to improve speed and usability without breaking the working Fooocus engine.

### Phase 1 — Safe Performance and UX Improvements

Phase 1 should be low risk and mostly reversible. These updates should not change the diffusion math, sampler internals, VAE behavior, model weights, or core generation semantics.

| Update | Goal | Risk | Notes |
| --- | --- | --- | --- |
| Preview frequency control | Reduce preview overhead during sampling | Low | Add options such as full preview, every 4 steps, every 8 steps, final only, disabled. |
| Smarter CLIP cache clearing | Avoid unnecessary prompt re-encoding | Low to Medium | Clear cache only when model, CLIP, clip-skip, or CLIP-affecting LoRA changes. |
| Draft/Fast/Balanced/Final presets | Make speed choices easier | Low | Use existing speed modes first before adding new engine behavior. |
| Refiner off by default in fast modes | Reduce extra generation work | Low | Keep refiner available for Final/Quality mode. |
| Optional metadata and image logging by mode | Faster draft iterations | Low | Draft mode can skip metadata by default; final mode can preserve it. |
| Better image format defaults | Faster saves for drafts | Low | Draft can use JPEG/WebP; final can use PNG. |
| LoRA cache key cleanup | Avoid unnecessary reloads | Low to Medium | Use stable keys based on normalized path, weight, mtime/hash, and model. |

### Phase 2 — Engine Hardening

Phase 2 can still be done safely, but it touches more important runtime structure. These changes should be isolated, tested, and feature-flagged where possible.

| Update | Goal | Risk | Notes |
| --- | --- | --- | --- |
| Split `refresh_everything()` | Avoid full refresh when only small inputs change | Medium | Keep the original wrapper as fallback until new refresh paths are proven. |
| Thread-safe job queue | Replace global list polling with safer task handling | Medium | Keep the existing handler behavior first; only change the task transport layer. |
| ControlNet/IP preprocessing cache | Speed retries with same input image/settings | Low to Medium | Cache by image hash, dimensions, preprocessor type, and thresholds. |
| Real async warmup | Hide text encoder/model prep latency | Medium | Must fall back to synchronous behavior if warmup fails. |
| Hardware profile selector | Make VRAM behavior easier | Low to Medium | Profiles: Low VRAM, Mid GPU, High VRAM, Cloud GPU. |
| Save/log pipeline cleanup | Reduce disk overhead in fast workflows | Low | Keep final-quality saving behavior intact. |

### Phase 3 — Advanced Redesign and Modernization

Phase 3 should only begin after Phase 1 and Phase 2 are validated.

| Update | Goal | Risk | Notes |
| --- | --- | --- | --- |
| Optional optimized attention/compile mode | Improve speed on compatible hardware | Medium to High | Keep experimental and opt-in. |
| Model warm pool | Faster switching between common model/LoRA combos | Medium | Must respect VRAM limits. |
| Better internal service boundaries | Make engine easier to maintain | Medium | Use the experimental redesign docs as guide. |
| Background research/agent workflow | Future AI-assisted engine analysis | Medium | Keep outside the critical sampler path. |
| Full UI redesign | Cleaner user experience | Medium | Preserve all existing power-user features behind Advanced mode. |

## Do Not Break These Paths

Before merging any engine or core changes, validate these generation paths:

- Text-to-image
- Multiple images / batch generation
- Seed increment and fixed seed
- Prompt expansion on/off
- Style selection
- Negative prompt behavior
- Vary subtle / vary strong
- Fast upscale
- Diffusion upscale
- Inpaint
- Outpaint
- Image Prompt
- FaceSwap
- PyraCanny
- CPDS
- Enhance mode
- Metadata save on/off
- Image log on/off
- PNG/JPEG/WebP output
- Low VRAM mode behavior
- Refiner disabled
- Refiner enabled

## Engine Safety Rules

1. Do not rewrite sampler math during Phase 1.
2. Do not upgrade pinned ML dependencies as the first move.
3. Do not change VAE encode/decode behavior without image regression checks.
4. Do not remove original Fooocus behavior until the replacement path is proven.
5. Keep old behavior behind fallback paths during refactors.
6. Use small branches, small commits, and clear rollback points.
7. Prefer feature flags and presets over hard behavior changes.
8. Treat `main` as stable and `develop` as the integration branch.

## Key Engine Areas to Improve

| Area | Current Concern | Future Direction |
| --- | --- | --- |
| `modules/default_pipeline.py` | Full refresh and aggressive cache clearing can waste time | Split refresh paths and preserve safe caches. |
| `modules/core.py` | Preview generation can add overhead during sampling | Add preview frequency controls and final-only mode. |
| `modules/async_worker.py` | Global task list and polling loop are fragile | Move to a thread-safe queue with explicit job state. |
| LoRA handling | Reload/rematch behavior can be expensive | Add better LoRA cache keys and reusable matched tensors. |
| ControlNet/IP preprocessing | Same image/settings may be recomputed | Add preprocessing cache. |
| Save/log pipeline | Draft runs can spend time on disk and metadata | Make output persistence mode-aware. |
| UI modes | Advanced options are powerful but overwhelming | Add Beginner/Pro and Draft/Fast/Balanced/Final presets. |

## Recommended Workflow

Use this process for all future work:

```text
main
  stable only

develop
  integration branch for tested work

feature branches
  perf/phase-1-fast-path
  engine/phase-2-core-hardening
  ui/preset-driven-workflow
  qa/regression-matrix
  docs/architecture-roadmap
```

Suggested merge order:

1. `docs/architecture-roadmap`
2. `qa/regression-matrix`
3. `perf/phase-1-fast-path`
4. `ui/preset-driven-workflow`
5. `engine/phase-2-core-hardening`
6. selected pieces from `experimental/full-system-redesign`

## Original Fooocus Context

Fooocus is an image generation software based on Gradio and Stable Diffusion XL. The upstream Fooocus project is in limited long-term support mode, focused primarily on bug fixes. This MiamiCreme fork is intended to explore improvements while respecting the existing working engine.

For original Fooocus usage, installation notes, model expectations, and upstream project context, refer to the original project:

- Upstream Fooocus: https://github.com/lllyasviel/Fooocus
- Current fork: https://github.com/miamicreme/Fooocus
- Experimental redesign branch: https://github.com/miamicreme/Fooocus/tree/experimental/full-system-redesign

## Installation

The standard Fooocus install flow still applies. Most users should follow the original Fooocus setup instructions unless this fork adds a specific new installer or preset workflow.

Basic local flow:

```bash
git clone https://github.com/miamicreme/Fooocus.git
cd Fooocus
python -m venv fooocus_env
source fooocus_env/bin/activate  # Linux/Mac
pip install -r requirements_versions.txt
python entry_with_update.py
```

Windows users can continue using the normal Fooocus Windows launcher flow if the packaged files are available.

## Roadmap Summary

This fork should become faster, cleaner, and easier to operate by focusing on:

- Fast defaults instead of manual tuning.
- Safer cache behavior.
- Better preview controls.
- Clear presets.
- A more reliable task queue.
- Better architecture documentation.
- Strong validation before touching the sampler core.

The goal is not to break Fooocus. The goal is to modernize around the working engine first, then redesign only after the system is fully mapped and validated.
