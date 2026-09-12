# v6 Working Execution Plan Alignment

Source: `Fooocus_AI_Image_Studio_Unified_Master_Plan_v6.pdf`

This document supersedes the earlier full-system-first roadmap order. The repo should now follow the v6 working execution rule:

```text
Studio first.
Engine second.
Adapter third.
Gallery/history fourth.
Guardrails last after scope.
```

The purpose is to keep the currently working local Fooocus engine stable while making AI Image Studio the separate control/planning layer.

---

## Final Execution Decision

| Area | v6 Position | Repo Action |
|---|---|---|
| AI Studio vs Fooocus UI | Separate AI Studio control layer | Do not add new helper panels into `webui.py` |
| Fooocus role | Stable SDXL engine/provider | Keep generation authority in Fooocus |
| Source of truth | `studio_knowledge.py` feature map | Planner routes through declarative feature data |
| Guardrails | Postponed until scope exists | No active blocking/warning/env toggles now |
| Generation strategy | One focused shot per run | Prompt planner creates focused shot prompts |
| Engine optimization | Only after Studio handoff works | No sampler/model lifecycle edits yet |
| Adapter | Thin contract first, execution later | Manual handoff/contract before direct calls |

---

## Current Known-Good Local Context

The v6 plan defines this as the accurate working context:

```text
Python 3.10.11
Torch 2.1.0+cu121
CUDA available
RTX 2060 6GB
Fooocus: http://localhost:7865
AI Studio: http://127.0.0.1:7872
```

Do not upgrade Gradio or other pinned dependencies without a dedicated dependency plan.

---

## Layer Boundaries

| Layer | Owns | Must Not Own Yet |
|---|---|---|
| AI Image Studio | User intent, feature routing, prompt planning, shot planning, handoff recipe | Sampler math, VAE internals, package versions, Fooocus model lifecycle |
| Fooocus Engine | Generation workflows, saves, model execution | Product planner UX or agent decisions |
| Adapter Layer | Converts Studio jobs into provider calls after planner stability | Deep engine rewrites or fragile-only Gradio function-index coupling |
| Guardrails Phase | Later policy scope and job-boundary enforcement | Active policy behavior during working phase |

---

## Immediate Branch Order

Follow this order before engine or adapter work:

```text
cleanup/no-webui-patch
studio/v2-planner
studio/no-guardrails-yet
studio/feature-map
studio/planner-tests
studio/prompt-quality
studio/ui-polish
```

Only after those pass:

```text
perf/phase-1-presets-preview
perf/phase-1-cache-defaults
engine/phase-2-refresh-split
engine/phase-2-queue
adapter/job-contract
studio/gallery-history
policy/guardrails-scope
```

---

## Global Failure Protocol

| Event | Required Action | Do Not Do |
|---|---|---|
| Compile fails | Fix on same branch; do not merge | Do not skip because UI appears to start |
| Smoke test fails | Stop next task and diagnose minimal diff | Do not begin engine or adapter work |
| Generation path regresses | Revert branch or restore old behavior behind fallback flag | Do not broadly patch sampler/handler |
| Engine task causes race/hang | Disable new path by default and keep old path | Do not merge until stress passes |
| Adapter timeout/error path unclear | Tighten contract first | Do not couple to unstable Gradio indexes |
| Prompt quality fails | Improve templates and rerun scorecard | Do not count routing test as full pass |

Failed validation blocks the next task.

---

## Task 1: Clean Old WebUI Patching Path

Branch:

```text
cleanup/no-webui-patch
```

Objective:

- Make Fooocus launch cleanly with no experimental panel and no automatic UI patching.
- Keep AI Studio separate from Fooocus UI.

Hard constraints:

- Do not add UI to `webui.py`.
- Do not modify generation logic, sampler, VAE, model management, dependencies, outputs, or model files.

Acceptance criteria:

- `webui.py` compiles.
- Old helper panel does not appear.
- `scripts/remove_easy_sdxl_webui.py` can run twice safely.
- Fooocus still starts.

Verification commands:

```powershell
python scripts\remove_easy_sdxl_webui.py
python -m py_compile webui.py
.\RUN_FOOOCUS_LOCAL.bat
```

---

## Task 2: Activate AI Studio v2 Planner

Branch:

```text
studio/v2-planner
```

Objective:

- Wire `ai_studio_app.py` to `local_markup.ai_studio_agent_v2`.
- Verify Studio returns selected feature, prompt, shots, and handoff recipe.

Expected one-line code change:

```python
# Old
from local_markup.ai_studio_agent import build_agent_outputs

# New
from local_markup.ai_studio_agent_v2 import build_agent_outputs
```

Hard constraints:

- No guardrails.
- No adapter.
- No direct generation.
- No `webui.py` patching.
- No port changes unless documented.

Verification commands:

```powershell
python -m py_compile ai_studio_app.py local_markup\ai_studio_agent_v2.py local_markup\studio_knowledge.py
.\RUN_AI_STUDIO.bat
```

---

## Task 3: Remove Active Guardrail Dependency

Branch:

```text
studio/no-guardrails-yet
```

Objective:

- Make `ai_studio_agent_v2` independent of `studio_guardrails.py`.

Hard constraints:

- Do not delete `studio_guardrails.py`.
- Do not add policy, blocking, warnings, env toggles, or logging.

Acceptance criteria:

- Planner output has no Guardrails section.
- Planner has no guardrail import or call.

Verification commands:

```powershell
python -m py_compile local_markup\ai_studio_agent_v2.py local_markup\studio_knowledge.py
```

---

## Task 4: Complete Source-of-Truth Feature Map

Branch:

```text
studio/feature-map
```

Required features:

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

Each feature needs:

```text
key
label
fooocus_area
use_when
avoid_when
required_inputs
outputs
controls
notes
```

Hard constraints:

- No UI code.
- No guardrails.
- No Gradio/Torch/Fooocus runtime imports.

---

## Task 5: Planner Scenario Tests

Branch:

```text
studio/planner-tests
```

Test cases:

```python
TEST_CASES = [
    ("Create a clean product photo of a sneaker", "text_to_image"),
    ("Use this image as style inspiration", "image_prompt"),
    ("Make a standing lifestyle photo from this source", "face_reference"),
    ("Remove the jacket from this photo", "inpaint"),
    ("Use this pose but make a new image", "pyracanny"),
    ("Follow this composition but not every edge", "cpds"),
    ("Make this bigger and sharper", "upscale"),
    ("Make a similar but different version", "variation"),
    ("Describe this image as a prompt", "describe"),
]
```

Verification command:

```powershell
python -m pytest tests/test_studio_planner.py -q
```

---

## Task 6: Prompt Quality Scorecard

Branch:

```text
studio/prompt-quality
```

Prompt quality criteria:

| Criterion | Pass Standard |
|---|---|
| Specificity | One shot, one setting, one task, one desired outcome |
| Feature fit | Prompt matches selected Fooocus workflow |
| Reference handling | Prompt explains how source/reference should influence output |
| Negative prompt | Blocks common defects without contradicting desired output |
| Handoff clarity | User knows exactly which Fooocus tab/control to use |

Routing tests are not prompt-quality tests. Both must pass.

---

## Task 7: Studio UI Clarity Pass

Branch:

```text
studio/ui-polish
```

Objective:

- Group outputs into selected feature, Fooocus area, best first prompt, negative prompt, shots, handoff, and next steps.

Hard constraints:

- No adapter.
- No direct generation.
- No guardrails.
- No engine changes.
- No dependency upgrades.

---

## Do Not Touch Yet

```text
sampler math
sample_hijack behavior
VAE encode/decode internals
model_management internals
Brownian noise sampler behavior
inpaint latent math
refiner swap math
pinned package versions
guardrail blocking/warning logic
```

---

## Adapter Comes Later

The adapter task must start with a contract only:

```text
ImageStudioJob
ReferenceImage
ProviderAdapter protocol
JobStatus states
AdapterError types
timeout/cancel semantics
ManualHandoffAdapter
```

`LocalFooocusAdapter` should raise `NotImplementedError` until contract tests pass.

---

## Final Rule

Do not move to engine hardening, adapter execution, gallery/history, or guardrails until Studio planner, feature map, scenario tests, and prompt-quality scorecard pass.
