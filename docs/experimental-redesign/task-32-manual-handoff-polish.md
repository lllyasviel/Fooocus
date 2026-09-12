# Task 32: Manual Handoff Polish

Branch: `studio/manual-handoff-polish`

## Goal

Make the Studio language clear enough that a user can move from idea to first Fooocus generation without guessing.

## What Changed

### No-friction usage guide

Added:

```text
local_markup/studio_usage_guide.py
tests/test_studio_usage_guide.py
```

The guide explains:

1. start Fooocus,
2. open Fooocus locally,
3. start AI Studio,
4. describe one image,
5. upload only needed references,
6. build the Fooocus plan,
7. copy the fields in order,
8. generate one image first,
9. review before continuing.

### UI copy polish

Updated:

```text
ai_studio_app.py
```

The Studio Agent now uses clearer labels:

- `What image do you want?`
- `Build My Fooocus Plan`
- `Step 1: Copy this prompt`
- `Step 2: Copy this negative prompt`
- `Step 3: Follow these Fooocus setup steps`
- `Review before generating`

### Output language polish

Updated:

```text
local_markup/studio_workflow_controller.py
```

The adapter preview now reads as:

- `Ready for Fooocus`,
- workflow,
- Fooocus area,
- hardware recommendation,
- references to upload,
- simple setup steps,
- why this setup.

## Validation Commands

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_workflow_controller.py local_markup\studio_usage_guide.py
python -m pytest tests/test_studio_usage_guide.py tests/test_studio_workflow_controller.py tests/test_ai_studio_app_foundation_preview.py -q
```

Full safe suite:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py tests/test_engine_hardware_profiles.py tests/test_studio_provider_registry.py tests/test_local_fooocus_adapter.py tests/test_studio_adapter_mappings.py tests/test_studio_generation_history.py tests/test_studio_hardware_preview.py tests/test_studio_adapter_preview.py tests/test_studio_history_preview.py tests/test_studio_manual_submit_package.py tests/test_studio_queue_dry_run.py tests/test_studio_completion_audit.py tests/test_studio_workflow_controller.py tests/test_ai_studio_app_foundation_preview.py tests/test_studio_worker_bridge_plan.py tests/test_studio_usage_guide.py -q
```

## Runtime Boundary

No live generation path changed. This branch is language, instruction, and handoff polish only.
