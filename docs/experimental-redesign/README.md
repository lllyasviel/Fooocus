# Experimental Redesign Index

This folder documents the plan for rebuilding the current Fooocus repo into a modular, API-first AI image studio while preserving existing functionality.

## Documents

| File | Purpose |
|---|---|
| `00-experimental-branch-plan.md` | Branch rules, migration strategy, success criteria |
| `01-function-map-and-inventory.md` | Current subsystem map and function inventory plan |
| `03-target-architecture.md` | Target platform architecture and package layout |
| `04-build-roadmap.md` | Task branches, phases, and acceptance gates |

## Tooling

A function inventory generator was added at:

```text
tools/architecture/function_inventory.py
```

Run it from the repo root:

```bash
python tools/architecture/function_inventory.py \
  --root . \
  --out docs/experimental-redesign/generated-function-inventory.md \
  --json docs/experimental-redesign/generated-function-inventory.json
```

The generated inventory should be committed in a follow-up branch after review.

## Recommended Next Branches

```text
feat/generation-request-schema
refactor/fooocus-engine-adapter
feat/api-job-service
feat/queue-worker-runtime
feat/web-studio-shell
```

## Merge Rule

Do not merge experimental code to `main` until parity tests prove the new path can replicate the current user-facing workflows.
