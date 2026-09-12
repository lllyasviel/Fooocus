from __future__ import annotations

from local_markup.engine_phase1_presets import (
    GENERATION_PRESETS,
    PreviewMode,
    get_generation_preset,
    list_generation_preset_keys,
    should_emit_preview,
)


def test_preview_mode_full_emits_every_step() -> None:
    assert all(should_emit_preview(step, 10, PreviewMode.FULL) for step in range(10))


def test_preview_mode_disabled_emits_no_steps() -> None:
    assert not any(should_emit_preview(step, 10, PreviewMode.DISABLED) for step in range(10))


def test_preview_mode_final_only_emits_final_step() -> None:
    emitted = [step for step in range(10) if should_emit_preview(step, 10, PreviewMode.FINAL_ONLY)]

    assert emitted == [9]


def test_preview_mode_every_four_emits_interval_and_final() -> None:
    emitted = [step for step in range(10) if should_emit_preview(step, 10, PreviewMode.EVERY_4)]

    assert emitted == [0, 4, 8, 9]


def test_preview_mode_every_eight_emits_interval_and_final() -> None:
    emitted = [step for step in range(10) if should_emit_preview(step, 10, PreviewMode.EVERY_8)]

    assert emitted == [0, 8, 9]


def test_generation_presets_have_expected_keys() -> None:
    assert list_generation_preset_keys() == ["draft", "fast", "balanced", "detail", "final"]


def test_generation_presets_keep_final_full_preview_available() -> None:
    final = get_generation_preset("final")

    assert final.preview_mode == PreviewMode.FULL
    assert final.refiner_default == "current_default"
    assert set(GENERATION_PRESETS) == {"draft", "fast", "balanced", "detail", "final"}
