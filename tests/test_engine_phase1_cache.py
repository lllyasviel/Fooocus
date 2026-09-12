from __future__ import annotations

from pathlib import Path

from local_markup.engine_phase1_cache import (
    LoraCacheKey,
    build_clip_cache_key,
    build_lora_cache_key,
    get_runtime_defaults_profile,
    should_invalidate_clip_cache,
)


def test_lora_cache_key_uses_structured_fields() -> None:
    key = build_lora_cache_key("models/loras/example.safetensors", 0.75, "base.safetensors", mtime_ns=123)

    assert Path(key.path).parts[-3:] == ("models", "loras", "example.safetensors")
    assert key.weight == 0.75
    assert key.mtime_ns == 123
    assert key.base_model == "base.safetensors"


def test_clip_cache_key_is_order_stable_for_loras() -> None:
    a = LoraCacheKey("b.safetensors", 1.0, 20, "base.safetensors")
    b = LoraCacheKey("a.safetensors", 0.5, 10, "base.safetensors")

    key_one = build_clip_cache_key("base.safetensors", 2, [a, b])
    key_two = build_clip_cache_key("base.safetensors", 2, [b, a])

    assert key_one == key_two


def test_clip_cache_invalidates_on_first_use_and_changes() -> None:
    key = build_clip_cache_key("base.safetensors", 1, [])
    same_key = build_clip_cache_key("base.safetensors", 1, [])
    changed_clip_skip = build_clip_cache_key("base.safetensors", 2, [])

    assert should_invalidate_clip_cache(None, key) is True
    assert should_invalidate_clip_cache(key, same_key) is False
    assert should_invalidate_clip_cache(key, changed_clip_skip) is True


def test_runtime_defaults_keep_legacy_available() -> None:
    legacy = get_runtime_defaults_profile("legacy")
    draft = get_runtime_defaults_profile("draft")
    final = get_runtime_defaults_profile("final")

    assert legacy.metadata_mode == "current_default"
    assert legacy.refiner_default == "current_default"
    assert draft.refiner_default == "off"
    assert final.metadata_mode == "full"
