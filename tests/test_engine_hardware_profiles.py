from __future__ import annotations

from local_markup.engine_hardware_profiles import HardwareTier, profile_summary, select_hardware_profile
from local_markup.engine_phase1_presets import PreviewMode


def test_selects_low_vram_profile_for_rtx_2060_size() -> None:
    profile = select_hardware_profile(6)

    assert profile.key == "low_vram_6gb"
    assert profile.tier == HardwareTier.LOW_VRAM
    assert profile.preview_mode == PreviewMode.FINAL_ONLY
    assert profile.recommended_preset == "fast"


def test_selects_mid_and_high_vram_profiles() -> None:
    assert select_hardware_profile(8).key == "mid_vram_8_12gb"
    assert select_hardware_profile(12).key == "mid_vram_8_12gb"
    assert select_hardware_profile(16).key == "high_vram_16gb_plus"


def test_profile_summary_is_copy_ready() -> None:
    summary = profile_summary(select_hardware_profile(6))

    assert "low_vram_6gb" in summary
    assert "preset=fast" in summary
    assert "previews=final_only" in summary
