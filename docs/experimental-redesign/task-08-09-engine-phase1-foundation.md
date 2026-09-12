# Tasks 8-9: Phase 1 Engine Foundation

Branch stack:

```text
perf/phase-1-presets-preview
perf/phase-1-cache-defaults
```

These tasks add testable helper modules for presets, preview cadence, cache keys, and default profiles.

## Task 8 Files

```text
local_markup/engine_phase1_presets.py
tests/test_engine_phase1_presets.py
```

Adds:

- `PreviewMode`
- `should_emit_preview()`
- `GenerationPreset`
- Draft/Fast/Balanced/Detail/Final preset data

## Task 9 Files

```text
local_markup/engine_phase1_cache.py
tests/test_engine_phase1_cache.py
```

Adds:

- `LoraCacheKey`
- `ClipCacheKey`
- stable LoRA key creation
- order-stable CLIP cache key creation
- cache invalidation helper
- runtime default profiles with legacy behavior still available

## Runtime Boundary

These helpers are not connected to the active sampler, queue, model lifecycle, or progress handler yet. That is intentional. The goal is to make the policy testable before wiring it into the engine.

## Verification Commands

```powershell
python -m py_compile local_markup\engine_phase1_presets.py local_markup\engine_phase1_cache.py
python -m pytest tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py -q
```

## Later Wiring Gate

Do not wire these helpers into active engine paths until:

- Studio planner tests pass,
- prompt review tests pass,
- old local generation still works,
- manual validation confirms old settings can be restored.
