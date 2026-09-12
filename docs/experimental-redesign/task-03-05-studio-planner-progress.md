# Studio Planner Tasks 3-5 Progress

This branch stack continues the v6 working-system-first plan after `studio/activate-planner`.

## Task 3: Planner Clean Dependency

Branch:

```text
studio/planner-clean
```

Completed:

- Removed the active `studio_guardrails.py` import from `local_markup/ai_studio_agent_v2.py`.
- Removed the runtime validation call from `build_agent_plan()`.
- Removed the `guardrail_status` field from `AgentPlan`.
- Removed the Guardrails section from `AgentPlan.as_markdown()`.
- Kept `studio_guardrails.py` untouched for a later scope phase.

Validation command:

```powershell
python -m py_compile local_markup\ai_studio_agent_v2.py local_markup\studio_knowledge.py
```

## Task 4: Source-of-Truth Feature Map

Branch:

```text
studio/feature-map
```

Completed:

- Reworked `StudioFeature` to use the v6 `notes` field for capability notes only.
- Removed active risk/guardrail fields from the feature map.
- Added every required v6 feature key:

```text
text_to_image
image_prompt
face_reference
pyracanny
cpds
upscale
variation
inpaint
auto_mask_sam
auto_mask_u2net
enhance
describe
styles
models_loras
```

- Added `REQUIRED_FEATURE_KEYS`.
- Added `validate_feature_map()`.
- Tightened scenario rules for common planner tests.

Validation command:

```powershell
python -m py_compile local_markup\studio_knowledge.py
```

## Task 5: Planner Scenario Tests

Branch:

```text
studio/planner-tests
```

Completed:

- Added `tests/test_studio_planner.py`.
- Tests cover common routing scenarios from v6:
  - text-to-image
  - image prompt
  - face/subject reference
  - inpaint
  - PyraCanny
  - CPDS
  - upscale
  - variation
  - describe
- Tests also cover exact-edit switch, bundle switch, required feature map completeness, copy-ready output fields, and output shape.

Validation command:

```powershell
python -m pytest tests/test_studio_planner.py -q
```

## Stop Conditions

Do not continue to prompt quality or UI polish until these commands pass locally:

```powershell
python -m py_compile ai_studio_app.py local_markup\ai_studio_agent_v2.py local_markup\studio_knowledge.py
python -m pytest tests/test_studio_planner.py -q
```

## Next Task

```text
studio/prompt-quality
```
