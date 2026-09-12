from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class BridgeStepState(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class BridgeStep:
    order: int
    name: str
    state: BridgeStepState
    note: str


BRIDGE_PLAN: Tuple[BridgeStep, ...] = (
    BridgeStep(1, "Studio dry-run job", BridgeStepState.READY, "Studio can build a job and dry-run adapter result."),
    BridgeStep(2, "Queue record", BridgeStepState.READY, "Queue contract can create queued records without active worker execution."),
    BridgeStep(3, "History record", BridgeStepState.READY, "Dry-run result can be recorded in Studio history."),
    BridgeStep(4, "Manual package", BridgeStepState.READY, "Manual submit package is copy-ready for Fooocus handoff."),
    BridgeStep(5, "Worker bridge", BridgeStepState.DEFERRED, "Do not call modules.async_worker until a feature flag and rollback path exist."),
    BridgeStep(6, "Live generation", BridgeStepState.BLOCKED, "Needs local manual approval after bridge tests and UI review."),
)


def bridge_plan_markdown() -> str:
    lines = ["## Local Worker Bridge Plan", "", "| Order | Step | State | Note |", "|---:|---|---|---|"]
    for step in BRIDGE_PLAN:
        lines.append(f"| {step.order} | {step.name} | `{step.state.value}` | {step.note} |")
    return "\n".join(lines)
