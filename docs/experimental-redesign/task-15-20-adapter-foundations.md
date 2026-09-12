# Tasks 15-20: Hardware, Adapter, Mapping, and History Foundations

Branch stack:

```text
engine/hardware-profiles
studio/provider-adapter-contract
studio/local-fooocus-adapter-poc
studio/adapter-image-reference
studio/adapter-inpaint-path
studio/generation-history
```

## Task 15: Hardware Profiles

Added:

```text
local_markup/engine_hardware_profiles.py
tests/test_engine_hardware_profiles.py
```

Purpose:

- recommend conservative defaults by VRAM tier,
- include a 6GB profile for RTX 2060-class machines,
- do not change active runtime defaults yet.

## Task 16: Provider Adapter Contract

Added:

```text
local_markup/studio_provider_registry.py
tests/test_studio_provider_registry.py
```

Purpose:

- register available Studio providers,
- return manual handoff and local dry-run adapters,
- keep provider selection testable and explicit.

## Task 17: Local Fooocus Adapter Proof-of-Concept

Added:

```text
local_markup/local_fooocus_adapter.py
tests/test_local_fooocus_adapter.py
```

Purpose:

- convert a Studio job into a queued engine record,
- return adapter result metadata,
- avoid calling the active Fooocus worker.

## Task 18: Image Prompt / Face Reference Mapping

Added:

```text
local_markup/studio_adapter_mappings.py
tests/test_studio_adapter_mappings.py
```

Purpose:

- create adapter job mappings for image prompt workflows,
- create face-reference-specific metadata,
- preserve reference image order and roles.

## Task 19: Inpaint Path Mapping

Extended:

```text
local_markup/studio_adapter_mappings.py
tests/test_studio_adapter_mappings.py
```

Purpose:

- map source image and mask image into explicit roles,
- label the Fooocus area as Inpaint,
- preserve prompt and negative prompt fields.

## Task 20: Generation History

Added:

```text
local_markup/studio_generation_history.py
tests/test_studio_generation_history.py
```

Purpose:

- convert adapter results into Studio history items,
- preserve prompt, negative prompt, workflow, seed, references, and adapter metadata,
- use deterministic manual history IDs when no job id exists.

## Validation Commands

```powershell
python -m py_compile local_markup\engine_hardware_profiles.py local_markup\studio_provider_registry.py local_markup\local_fooocus_adapter.py local_markup\studio_adapter_mappings.py local_markup\studio_generation_history.py

python -m pytest tests/test_engine_hardware_profiles.py tests/test_studio_provider_registry.py tests/test_local_fooocus_adapter.py tests/test_studio_adapter_mappings.py tests/test_studio_generation_history.py -q
```

Full safe suite:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py tests/test_engine_hardware_profiles.py tests/test_studio_provider_registry.py tests/test_local_fooocus_adapter.py tests/test_studio_adapter_mappings.py tests/test_studio_generation_history.py -q
```

## Runtime Boundary

These tasks add foundations only. They do not connect adapter execution to the active Fooocus generation worker and do not change sampler, VAE, model loading, UI generation, or package behavior.
