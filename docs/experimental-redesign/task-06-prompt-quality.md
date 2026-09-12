# Task 6: Prompt Quality Scorecard

Branch: `studio/prompt-quality`

This task adds a manual prompt-quality review layer for Studio planner outputs.

## Why

Planner routing tests prove that the right Fooocus workflow was selected. They do not prove that the prompt is useful. This task adds a scorecard so prompt quality can be reviewed separately from route correctness.

## Added Files

```text
local_markup/studio_prompt_quality.py
tests/test_studio_prompt_quality.py
```

## Scorecard Criteria

| Criterion | Review Goal |
|---|---|
| Specificity | One shot, one setting, one task, one desired outcome |
| Feature fit | Prompt matches the selected Fooocus workflow |
| Reference handling | Prompt explains how source/reference should guide output |
| Negative prompt | Negative prompt blocks common defects without fighting the desired result |
| Handoff clarity | User knows exactly which Fooocus tab/control to use |

## Verification Commands

```powershell
python -m py_compile local_markup\ai_studio_agent_v2.py local_markup\studio_prompt_quality.py
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py -q
```

## Stop Conditions

Do not move to UI polish if:

- planner tests fail,
- prompt quality review tests fail,
- review markdown does not include all criteria,
- generated prompts are clearly vague or not copy-ready.
