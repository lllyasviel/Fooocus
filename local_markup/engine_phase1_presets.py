from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class PreviewMode(str, Enum):
    FULL = "full"
    EVERY_4 = "every_4"
    EVERY_8 = "every_8"
    FINAL_ONLY = "final_only"
    DISABLED = "disabled"


def should_emit_preview(step: int, total_steps: int, mode: PreviewMode) -> bool:
    """Return whether a preview should be emitted for a sampling step.

    This helper is intentionally not wired into the current Fooocus sampler yet.
    It gives the Phase 1 preview policy a testable, reversible shape before any
    engine runtime behavior changes.
    """

    if total_steps <= 0:
        return False
    step = max(0, step)
    final_step = step >= total_steps - 1

    if mode == PreviewMode.DISABLED:
        return False
    if mode == PreviewMode.FINAL_ONLY:
        return final_step
    if mode == PreviewMode.EVERY_4:
        return step % 4 == 0 or final_step
    if mode == PreviewMode.EVERY_8:
        return step % 8 == 0 or final_step
    return True


@dataclass(frozen=True)
class GenerationPreset:
    key: str
    label: str
    description: str
    performance: str
    default_steps: int | None
    preview_mode: PreviewMode
    metadata_mode: str
    refiner_default: str


GENERATION_PRESETS: Dict[str, GenerationPreset] = {
    "draft": GenerationPreset(
        key="draft",
        label="Draft",
        description="Fast idea check with fewer previews and lightweight logging.",
        performance="Speed",
        default_steps=None,
        preview_mode=PreviewMode.FINAL_ONLY,
        metadata_mode="minimal",
        refiner_default="off",
    ),
    "fast": GenerationPreset(
        key="fast",
        label="Fast",
        description="Quick candidate generation while keeping normal output behavior available.",
        performance="Speed",
        default_steps=None,
        preview_mode=PreviewMode.EVERY_8,
        metadata_mode="standard",
        refiner_default="current_default",
    ),
    "balanced": GenerationPreset(
        key="balanced",
        label="Balanced",
        description="Default Studio recommendation for usable candidates.",
        performance="Quality",
        default_steps=None,
        preview_mode=PreviewMode.EVERY_4,
        metadata_mode="standard",
        refiner_default="current_default",
    ),
    "detail": GenerationPreset(
        key="detail",
        label="Detail",
        description="Slower preview cadence for detail-oriented runs.",
        performance="Quality",
        default_steps=None,
        preview_mode=PreviewMode.EVERY_4,
        metadata_mode="full",
        refiner_default="current_default",
    ),
    "final": GenerationPreset(
        key="final",
        label="Final",
        description="Final candidate generation with old full-preview behavior available.",
        performance="Quality",
        default_steps=None,
        preview_mode=PreviewMode.FULL,
        metadata_mode="full",
        refiner_default="current_default",
    ),
}


def get_generation_preset(key: str) -> GenerationPreset:
    return GENERATION_PRESETS[key]


def list_generation_preset_keys() -> list[str]:
    return list(GENERATION_PRESETS)
