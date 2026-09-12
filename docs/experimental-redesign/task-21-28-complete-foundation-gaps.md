# Tasks 21-28: Complete Foundation Gaps

Branch: `studio/complete-foundation-gaps`

This stack closes the high-ROI gaps found after the Tasks 1-20 audit while keeping active Fooocus generation unchanged.

## Pass 1: History Reference Metadata

Status: complete.

History now records actual reference names, paths, and roles, not just reference count.

Files:

```text
local_markup/studio_generation_history.py
tests/test_studio_generation_history.py
```

## Pass 2: Hardware Preview

Status: complete.

Adds copy-ready hardware profile markdown for 6GB, mid-VRAM, and high-VRAM profiles.

Files:

```text
local_markup/studio_hardware_preview.py
tests/test_studio_hardware_preview.py
```

## Pass 3: Adapter Mapping Preview

Status: complete.

Adds markdown rendering for adapter jobs and mapping notes before any handoff.

Files:

```text
local_markup/studio_adapter_preview.py
tests/test_studio_adapter_preview.py
```

## Pass 4: Read-Only History Preview

Status: complete.

Adds markdown rendering for history items and history stores.

Files:

```text
local_markup/studio_history_preview.py
tests/test_studio_history_preview.py
```

## Pass 5: Manual Submit Package

Status: complete.

Creates one package containing workflow, prompt, negative prompt, references, metadata, preview markdown, and handoff steps.

Files:

```text
local_markup/studio_manual_submit_package.py
tests/test_studio_manual_submit_package.py
```

## Pass 6: Queue Dry-Run Integration

Status: complete as dry-run.

Submits a Studio job through the local dry-run adapter and records the result in Studio history. It does not start active generation.

Files:

```text
local_markup/studio_queue_dry_run.py
tests/test_studio_queue_dry_run.py
```

## Pass 7: Completion Audit Data

Status: complete.

Adds a structured audit table showing complete, foundation-only, needs-local-validation, and deferred items.

Files:

```text
local_markup/studio_completion_audit.py
tests/test_studio_completion_audit.py
```

## Pass 8: Documentation

Status: complete.

This file documents the gap closure and validation commands.

## Validation Commands

```powershell
python -m py_compile local_markup\studio_hardware_preview.py local_markup\studio_adapter_preview.py local_markup\studio_history_preview.py local_markup\studio_manual_submit_package.py local_markup\studio_queue_dry_run.py local_markup\studio_completion_audit.py

python -m pytest tests/test_studio_generation_history.py tests/test_studio_hardware_preview.py tests/test_studio_adapter_preview.py tests/test_studio_history_preview.py tests/test_studio_manual_submit_package.py tests/test_studio_queue_dry_run.py tests/test_studio_completion_audit.py -q
```

Full safe suite:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py tests/test_engine_hardware_profiles.py tests/test_studio_provider_registry.py tests/test_local_fooocus_adapter.py tests/test_studio_adapter_mappings.py tests/test_studio_generation_history.py tests/test_studio_hardware_preview.py tests/test_studio_adapter_preview.py tests/test_studio_history_preview.py tests/test_studio_manual_submit_package.py tests/test_studio_queue_dry_run.py tests/test_studio_completion_audit.py -q
```

## Runtime Boundary

These passes do not wire active Fooocus generation. They complete the missing foundation layer required before safe UI and worker integration.
