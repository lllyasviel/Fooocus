from __future__ import annotations

import ast
from pathlib import Path


DEFAULT_PIPELINE = Path("modules/default_pipeline.py")


EXPECTED_PHASES = [
    "_reset_final_model_handles",
    "_refresh_requested_models",
    "_apply_loras_and_validate",
    "_bind_final_model_handles",
    "_ensure_expansion_loaded",
    "_prepare_text_encoder_and_clear_cache",
]


def _source() -> str:
    return DEFAULT_PIPELINE.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _function_node(name: str) -> ast.FunctionDef:
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function: {name}")


def _called_functions(function_name: str) -> list[str]:
    node = _function_node(function_name)
    body_tree = ast.Module(body=node.body, type_ignores=[])
    calls: list[str] = []
    for child in ast.walk(body_tree):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return calls


def test_refresh_everything_uses_named_phase_functions_in_order() -> None:
    calls = _called_functions("refresh_everything")

    phase_positions = [calls.index(phase) for phase in EXPECTED_PHASES]

    assert phase_positions == sorted(phase_positions)


def test_refresh_split_keeps_public_refresh_signature() -> None:
    node = _function_node("refresh_everything")
    args = [arg.arg for arg in node.args.args]

    assert args == ["refiner_model_name", "base_model_name", "loras", "base_model_additional_loras", "use_synthetic_refiner", "vae_name"]


def test_refresh_split_preserves_model_refresh_order() -> None:
    calls = _called_functions("_refresh_requested_models")

    synthetic_index = calls.index("refresh_base_model")
    synthesize_index = calls.index("synthesize_refiner_model")
    normal_refiner_index = calls.index("refresh_refiner_model")
    normal_base_index = calls.index("refresh_base_model", synthetic_index + 1)

    assert synthetic_index < synthesize_index
    assert normal_refiner_index < normal_base_index


def test_refresh_split_keeps_lora_then_integrity_order() -> None:
    calls = _called_functions("_apply_loras_and_validate")

    assert calls.index("refresh_loras") < calls.index("assert_model_integrity")


def test_refresh_split_keeps_final_handle_bindings() -> None:
    source = ast.get_source_segment(_source(), _function_node("_bind_final_model_handles"))

    assert "final_unet = model_base.unet_with_lora" in source
    assert "final_clip = model_base.clip_with_lora" in source
    assert "final_vae = model_base.vae" in source
    assert "final_refiner_unet = model_refiner.unet_with_lora" in source
    assert "final_refiner_vae = model_refiner.vae" in source


def test_refresh_split_keeps_text_encoder_then_cache_clear_order() -> None:
    calls = _called_functions("_prepare_text_encoder_and_clear_cache")

    assert calls == ["prepare_text_encoder", "clear_all_caches"]
