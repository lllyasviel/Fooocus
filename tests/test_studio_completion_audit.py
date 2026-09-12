from __future__ import annotations

from local_markup.studio_completion_audit import COMPLETION_AUDIT, CompletionState, audit_markdown


def test_completion_audit_has_explicit_foundation_and_deferred_states() -> None:
    states = {item.state for item in COMPLETION_AUDIT}

    assert CompletionState.COMPLETE in states
    assert CompletionState.FOUNDATION in states
    assert CompletionState.DEFERRED in states


def test_audit_markdown_is_table() -> None:
    markdown = audit_markdown()

    assert "Completion Audit" in markdown
    assert "| Item | State | Note |" in markdown
    assert "Queue contract" in markdown
    assert "Runtime hardware defaults" in markdown
