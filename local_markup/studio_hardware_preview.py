from __future__ import annotations

from local_markup.engine_hardware_profiles import HardwareProfile, profile_summary, select_hardware_profile


def build_hardware_profile_preview(vram_gb: int) -> str:
    profile = select_hardware_profile(vram_gb)
    return hardware_profile_markdown(profile)


def hardware_profile_markdown(profile: HardwareProfile) -> str:
    return "\n".join(
        [
            f"## Hardware Profile: {profile.key}",
            "",
            f"Tier: `{profile.tier.value}`",
            f"Recommended preset: `{profile.recommended_preset}`",
            f"Default size: `{profile.default_width}x{profile.default_height}`",
            f"Preview mode: `{profile.preview_mode.value}`",
            "",
            profile.notes,
            "",
            f"Summary: {profile_summary(profile)}",
        ]
    )
