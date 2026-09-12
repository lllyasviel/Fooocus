from __future__ import annotations

from local_markup.studio_copy_controls import COPY_READY_OUTPUT_LABELS, copy_controls_summary


def test_copy_controls_summary_lists_copy_ready_outputs() -> None:
    summary = copy_controls_summary()

    assert len(COPY_READY_OUTPUT_LABELS) == 6
    assert "Copy-ready outputs" in summary
    assert "Step 1: Copy this prompt" in summary
    assert "Step 2: Copy this negative prompt" in summary
