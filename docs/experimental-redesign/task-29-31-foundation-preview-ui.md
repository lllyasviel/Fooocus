# Tasks 29-31: Foundation Preview UI and Bridge Plan

Branch: `studio/wire-foundation-preview-ui`

## Task 29: Foundation Preview UI

Status: complete.

`ai_studio_app.py` now routes the Studio Agent button through `build_studio_workflow_outputs()` and returns:

- agent plan,
- primary prompt,
- negative prompt,
- selected workflow,
- selected Fooocus area,
- shot prompts,
- handoff recipe,
- adapter preview,
- history preview.

It also exposes a VRAM profile selector so the Studio output can include hardware-aware recommendations.

## Task 30: Session History UI Foundation

Status: complete as read-only foundation.

The Studio Agent tab shows a local history preview generated from the dry-run workflow result. This is not persistent storage yet.

## Task 31: Local Worker Bridge Plan

Status: complete as plan only.

Added:

```text
local_markup/studio_worker_bridge_plan.py
tests/test_studio_worker_bridge_plan.py
```

The plan marks:

- Studio dry-run job: ready,
- Queue record: ready,
- History record: ready,
- Manual package: ready,
- Worker bridge: deferred,
- Live generation: blocked.

## Validation Commands

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_workflow_controller.py local_markup\studio_worker_bridge_plan.py
python -m pytest tests/test_studio_workflow_controller.py tests/test_ai_studio_app_foundation_preview.py tests/test_studio_worker_bridge_plan.py -q
```

Full safe suite:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py tests/test_engine_hardware_profiles.py tests/test_studio_provider_registry.py tests/test_local_fooocus_adapter.py tests/test_studio_adapter_mappings.py tests/test_studio_generation_history.py tests/test_studio_hardware_preview.py tests/test_studio_adapter_preview.py tests/test_studio_history_preview.py tests/test_studio_manual_submit_package.py tests/test_studio_queue_dry_run.py tests/test_studio_completion_audit.py tests/test_studio_workflow_controller.py tests/test_ai_studio_app_foundation_preview.py tests/test_studio_worker_bridge_plan.py -q
```

## Runtime Boundary

This branch wires the preview/dry-run foundation into the Studio UI. It does not connect to active Fooocus generation or `modules.async_worker`.
