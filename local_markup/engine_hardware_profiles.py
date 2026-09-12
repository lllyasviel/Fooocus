from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from local_markup.engine_phase1_presets import PreviewMode


class HardwareTier(str, Enum):
    LOW_VRAM = "low_vram"
    MID_VRAM = "mid_vram"
    HIGH_VRAM = "high_vram"


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    tier: HardwareTier
    min_vram_gb: int
    max_vram_gb: Optional[int]
    default_width: int
    default_height: int
    preview_mode: PreviewMode
    recommended_preset: str
    notes: str


HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "low_vram_6gb": HardwareProfile(
        key="low_vram_6gb",
        tier=HardwareTier.LOW_VRAM,
        min_vram_gb=0,
        max_vram_gb=6,
        default_width=768,
        default_height=1024,
        preview_mode=PreviewMode.FINAL_ONLY,
        recommended_preset="fast",
        notes="Conservative defaults for 6GB cards such as RTX 2060.",
    ),
    "mid_vram_8_12gb": HardwareProfile(
        key="mid_vram_8_12gb",
        tier=HardwareTier.MID_VRAM,
        min_vram_gb=7,
        max_vram_gb=12,
        default_width=1024,
        default_height=1024,
        preview_mode=PreviewMode.EVERY_8,
        recommended_preset="balanced",
        notes="Balanced defaults for common 8GB to 12GB GPUs.",
    ),
    "high_vram_16gb_plus": HardwareProfile(
        key="high_vram_16gb_plus",
        tier=HardwareTier.HIGH_VRAM,
        min_vram_gb=13,
        max_vram_gb=None,
        default_width=1024,
        default_height=1344,
        preview_mode=PreviewMode.EVERY_4,
        recommended_preset="detail",
        notes="Higher-quality defaults for 16GB+ GPUs.",
    ),
}


def select_hardware_profile(vram_gb: int) -> HardwareProfile:
    for profile in HARDWARE_PROFILES.values():
        if vram_gb >= profile.min_vram_gb and (profile.max_vram_gb is None or vram_gb <= profile.max_vram_gb):
            return profile
    return HARDWARE_PROFILES["low_vram_6gb"]


def profile_summary(profile: HardwareProfile) -> str:
    return (
        f"{profile.key}: {profile.default_width}x{profile.default_height}, "
        f"preset={profile.recommended_preset}, previews={profile.preview_mode.value}"
    )
