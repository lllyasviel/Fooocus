from __future__ import annotations


COPY_READY_OUTPUT_LABELS = (
    "Use this Fooocus workflow",
    "Open this Fooocus tab or area",
    "Step 1: Copy this prompt",
    "Step 2: Copy this negative prompt",
    "Step 3: Follow these Fooocus setup steps",
    "Use these only after the first result",
)


def copy_controls_summary() -> str:
    labels = "\n".join([f"- {label}" for label in COPY_READY_OUTPUT_LABELS])
    return "## Copy-ready outputs\n\nEvery output below has a copy button:\n\n" + labels
