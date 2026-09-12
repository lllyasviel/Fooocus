# Tasks 11-14: Queue, Adapter, History, Scope Foundations

Branch stack:

```text
engine/queue-contract
adapter/job-contract
studio/gallery-history
docs/scope-checklist
```

## Task 11: Queue Contract

Added:

```text
local_markup/engine_queue_contract.py
tests/test_engine_queue_contract.py
```

Purpose:

- define job request/status/progress records,
- provide immutable transition helpers,
- avoid changing active async worker behavior yet.

## Task 12: Adapter Job Contract

Added:

```text
local_markup/studio_adapter_contract.py
tests/test_studio_adapter_contract.py
```

Purpose:

- define Studio-to-engine job shape,
- define reference image records,
- define provider adapter protocol,
- provide a manual handoff adapter that does not start generation.

## Task 13: Gallery / History

Added:

```text
local_markup/studio_history.py
tests/test_studio_history.py
```

Purpose:

- define history item records,
- support latest/history filtering,
- support workflow filtering and favorites,
- avoid UI/storage wiring yet.

## Task 14: Guardrails Scope Checklist

Added:

```text
docs/experimental-redesign/task-14-scope-checklist.md
```

Purpose:

- define future guardrails scope only,
- no runtime enforcement,
- no adapter rejection,
- no prompt mutation.

## Validation Commands

```powershell
python -m py_compile local_markup\engine_queue_contract.py local_markup\studio_adapter_contract.py local_markup\studio_history.py
python -m pytest tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py -q
```

Full current safe suite:

```powershell
python -m pytest tests/test_studio_planner.py tests/test_studio_prompt_quality.py tests/test_engine_phase1_presets.py tests/test_engine_phase1_cache.py tests/test_engine_refresh_split.py tests/test_engine_queue_contract.py tests/test_studio_adapter_contract.py tests/test_studio_history.py -q
```

## Runtime Boundary

These tasks add contracts, records, helper behavior, and docs. They do not change active generation execution.
