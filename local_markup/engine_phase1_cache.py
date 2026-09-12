from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class LoraCacheKey:
    path: str
    weight: float
    mtime_ns: int
    base_model: str


@dataclass(frozen=True)
class ClipCacheKey:
    base_model: str
    clip_skip: int
    clip_affecting_loras: Tuple[LoraCacheKey, ...]


@dataclass(frozen=True)
class RuntimeDefaultsProfile:
    key: str
    metadata_mode: str
    refiner_default: str
    log_mode: str


RUNTIME_DEFAULT_PROFILES = {
    "legacy": RuntimeDefaultsProfile(
        key="legacy",
        metadata_mode="current_default",
        refiner_default="current_default",
        log_mode="current_default",
    ),
    "draft": RuntimeDefaultsProfile(
        key="draft",
        metadata_mode="minimal",
        refiner_default="off",
        log_mode="reduced",
    ),
    "final": RuntimeDefaultsProfile(
        key="final",
        metadata_mode="full",
        refiner_default="current_default",
        log_mode="full",
    ),
}


def file_mtime_ns(path: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return file_path.stat().st_mtime_ns


def build_lora_cache_key(path: str, weight: float, base_model: str, mtime_ns: int | None = None) -> LoraCacheKey:
    return LoraCacheKey(
        path=str(Path(path)),
        weight=float(weight),
        mtime_ns=file_mtime_ns(path) if mtime_ns is None else int(mtime_ns),
        base_model=base_model,
    )


def build_clip_cache_key(base_model: str, clip_skip: int, loras: Iterable[LoraCacheKey]) -> ClipCacheKey:
    return ClipCacheKey(
        base_model=base_model,
        clip_skip=int(clip_skip),
        clip_affecting_loras=tuple(sorted(loras, key=lambda item: (item.path, item.weight, item.mtime_ns, item.base_model))),
    )


def should_invalidate_clip_cache(previous: ClipCacheKey | None, current: ClipCacheKey) -> bool:
    if previous is None:
        return True
    return previous != current


def get_runtime_defaults_profile(key: str) -> RuntimeDefaultsProfile:
    return RUNTIME_DEFAULT_PROFILES[key]
