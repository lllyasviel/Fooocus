from __future__ import annotations

from local_markup.studio_hardware_preview import build_hardware_profile_preview


def test_build_hardware_profile_preview_for_6gb_machine() -> None:
    markdown = build_hardware_profile_preview(6)

    assert "Hardware Profile: low_vram_6gb" in markdown
    assert "Recommended preset: `fast`" in markdown
    assert "Preview mode: `final_only`" in markdown
    assert "RTX 2060" in markdown
