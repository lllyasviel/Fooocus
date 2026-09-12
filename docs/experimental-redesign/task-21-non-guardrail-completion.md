# Task 21: Non-Guardrail Completion

Branch: `studio/non-guardrail-wiring`

This branch finishes the non-guardrail Studio wiring by connecting the planner, adapter mapping, local dry-run adapter, hardware profile recommendation, and history preview into the AI Studio UI.

## Completed

- Studio planner still produces the primary prompt, negative prompt, selected workflow, selected Fooocus area, shot prompts, and handoff recipe.
- Studio workflow controller now builds a complete local workflow run.
- Adapter mapping now converts plans into `ImageStudioJob` records.
- Image prompt, face-reference, and inpaint paths map references into explicit roles.
- Local dry-run adapter now accepts the Studio job and returns an adapter result.
- Hardware profile recommendation is included in the adapter preview.
- Studio history item is created from the adapter result.
- History now stores full reference metadata, not only reference count.
- AI Studio UI now shows adapter preview and history preview after planning.

## Still Not Included

Guardrails are intentionally excluded from this branch.

Direct active Fooocus worker submission is also not enabled in this branch. The UI uses the local dry-run adapter to prepare and verify job shape while keeping Fooocus generation manual and separate.

## Why Active Worker Submission Is Separate

The active Fooocus worker has global runtime state, GPU memory behavior, model lifecycle concerns, and queue behavior that should be validated in a dedicated runtime branch. This branch finishes the Studio-side workflow without changing sampler, VAE, model loading, async worker, or package behavior.

## Validation

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_workflow_controller.py local_markup\studio_generation_history.py

python -m pytest tests/test_studio_workflow_controller.py tests/test_studio_generation_history.py -q
```

Full safe suite:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py tests/test_engine_hardware_profiles.py tests/test_studio_provider_registry.py tests/test_local_fooocus_adapter.py tests/test_studio_adapter_mappings.py tests/test_studio_generation_history.py tests/test_studio_workflow_controller.py -q
```

## Manual UI Check

```powershell
.\RUN_AI_STUDIO.bat
```

Expected:

- Studio plan appears.
- Copy-ready prompt fields appear.
- Adapter preview appears.
- History preview appears.
- No guardrail enforcement appears.
- No active Fooocus generation starts automatically.
