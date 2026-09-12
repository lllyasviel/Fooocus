# Task 14: Guardrails Scope Checklist

Branch: `docs/scope-checklist`

This task defines the scope for future Studio guardrails without adding runtime enforcement.

## Current Rule

Guardrails remain last in the v6 order.

```text
Studio first.
Engine second.
Adapter third.
Gallery/history fourth.
Guardrails last after scope.
```

## What This Task Adds

A planning checklist only. No active blocking, warnings, prompt mutation, logging, adapter rejection, or generation filtering is added here.

## Future Guardrails Scope

Future implementation should answer these questions before any runtime behavior is changed:

| Area | Decision Needed |
|---|---|
| User intent | What user requests should require review before handoff? |
| Reference images | How should identity/reference uncertainty be explained? |
| Output limits | Which limits belong in Studio copy versus engine behavior? |
| Adapter behavior | Should the adapter reject jobs or return manual review steps? |
| History | Should rejected/manual-review jobs appear in history? |
| UX | Should messaging be informational, blocking, or both? |
| Config | Which settings should be configurable per local/user deployment? |

## Non-Goals For This Branch

- No changes to `local_markup/studio_guardrails.py`.
- No new enforcement in `ai_studio_agent_v2.py`.
- No active checks in the adapter contract.
- No changes to Fooocus image filtering.
- No changes to sampler, queue, model loading, or UI generation behavior.

## Acceptance Criteria Before Future Runtime Work

Before a future guardrails runtime branch starts, create a short design note with:

1. exact reviewed scenarios,
2. exact user-facing copy,
3. whether behavior is informational or blocking,
4. how users can recover or revise the job,
5. tests that prove normal Studio planning still works.
