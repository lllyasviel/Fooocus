# Task 7: AI Studio UI Clarity Pass

Branch: `studio/ui-polish`

This task improves AI Studio copy/paste clarity without adding adapter execution or changing Fooocus engine behavior.

## What Changed

`ai_studio_app.py` now groups the Studio Agent tab into clearer steps:

1. Describe the goal.
2. Copy selected workflow and Fooocus tab.
3. Copy best first shot prompt.
4. Copy negative prompt.
5. Copy handoff recipe.
6. Use next shot prompts after the first result.
7. Expand full agent plan when needed.

## Hard Boundaries Kept

- No adapter.
- No direct generation.
- No engine changes.
- No dependency changes.
- No Fooocus `webui.py` patching.

## Verification Commands

```powershell
python -m py_compile ai_studio_app.py
.\RUN_AI_STUDIO.bat
```

## Manual UI Scenarios

Check that fields are readable and copy-ready for:

- new image
- reference image
- local edit
- pose control
- upscale

## Stop Conditions

Do not continue if the UI hides the handoff recipe, changes output count/order, or breaks the existing click binding.
