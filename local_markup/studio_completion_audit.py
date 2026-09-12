from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class CompletionState(str, Enum):
    COMPLETE = "complete"
    FOUNDATION = "foundation"
    NEEDS_LOCAL_VALIDATION = "needs_local_validation"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class CompletionAuditItem:
    name: str
    state: CompletionState
    note: str


COMPLETION_AUDIT: Tuple[CompletionAuditItem, ...] = (
    CompletionAuditItem("Fooocus launch baseline", CompletionState.COMPLETE, "Validated locally after refresh split."),
    CompletionAuditItem("Studio planner", CompletionState.COMPLETE, "Planner routing and copy-ready outputs are tested."),
    CompletionAuditItem("Prompt quality", CompletionState.COMPLETE, "Manual review criteria are tested."),
    CompletionAuditItem("Refresh split", CompletionState.COMPLETE, "Public refresh function remains stable with phase tests."),
    CompletionAuditItem("Local URL runner", CompletionState.NEEDS_LOCAL_VALIDATION, "Patch exists; local machine should confirm 127.0.0.1 output."),
    CompletionAuditItem("Queue contract", CompletionState.FOUNDATION, "Records and transitions exist; active worker wiring is deferred."),
    CompletionAuditItem("Adapter contract", CompletionState.FOUNDATION, "Manual and dry-run paths exist; live generation is deferred."),
    CompletionAuditItem("History and previews", CompletionState.FOUNDATION, "Records and markdown previews exist; persistent UI wiring is deferred."),
    CompletionAuditItem("Runtime hardware defaults", CompletionState.DEFERRED, "Profiles exist but are not applied automatically yet."),
)


def audit_markdown() -> str:
    lines = ["## Studio/Fooocus Completion Audit", "", "| Item | State | Note |", "|---|---|---|"]
    for item in COMPLETION_AUDIT:
        lines.append(f"| {item.name} | `{item.state.value}` | {item.note} |")
    return "\n".join(lines)
