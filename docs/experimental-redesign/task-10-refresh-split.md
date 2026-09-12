# Task 10: Engine Phase 2 Refresh Split

Branch: `engine/phase-2-refresh-split`

This task splits `modules/default_pipeline.py::refresh_everything()` into named phases while preserving the old public function and call order.

## Goal

Make the model refresh path easier to reason about before later queue or runtime work.

## What Changed

`refresh_everything()` now delegates to these phases:

```text
_reset_final_model_handles
_refresh_requested_models
_apply_loras_and_validate
_bind_final_model_handles
_ensure_expansion_loaded
_prepare_text_encoder_and_clear_cache
```

## Behavior Kept

The previous sequence is preserved:

```text
1. clear final handles
2. refresh requested refiner/base model path
3. refresh LoRAs
4. assert model integrity
5. bind final runtime handles
6. ensure expansion exists
7. prepare text encoder
8. clear prompt cache
```

## Four-Pass Review

### Pass 1: Implementation

- Split refresh into private phase helpers.
- Kept the public `refresh_everything()` signature.
- Kept default startup call unchanged.

### Pass 2: Source Tests

Added `tests/test_engine_refresh_split.py`.

The tests avoid importing `modules.default_pipeline` because that module loads engine state at import time.

Test coverage checks:

- phase functions exist,
- public signature is preserved,
- phase order is preserved,
- synthetic branch still loads base before synthetic refiner,
- normal branch still refreshes refiner before base,
- LoRA refresh still happens before integrity assertion,
- final handle binding remains the same,
- text encoder preparation still happens before cache clear.

### Pass 3: Compatibility Review

Diff scope is intentionally narrow:

```text
modules/default_pipeline.py
tests/test_engine_refresh_split.py
docs/experimental-redesign/task-10-refresh-split.md
```

No sampler, VAE, scheduler, inpaint, queue, dependency, or UI behavior is changed.

### Pass 4: Local Validation Required

Run:

```powershell
python -m py_compile modules\default_pipeline.py
python -m pytest tests/test_engine_refresh_split.py -q
```

Then run the existing safe suite from the previous branch:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py -q
```

## Stop Conditions

Do not continue to queue/runtime work if:

- `default_pipeline.py` fails to compile,
- source tests fail,
- Fooocus fails during local startup,
- generated images differ because of this refactor,
- any sampler/process diffusion code changed unexpectedly.
