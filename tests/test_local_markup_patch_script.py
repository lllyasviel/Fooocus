from pathlib import Path
import importlib.util


def load_patch_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "apply_local_markup_ui_patch.py"
    spec = importlib.util.spec_from_file_location("apply_local_markup_ui_patch", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_script_contains_expected_markers():
    module = load_patch_module()
    assert "local_markup_instruction" in module.NEW_WIDGET_BLOCK
    assert "local_markup_button.click" in module.NEW_CLICK_BLOCK
    assert "build_markup_ui_outputs" in module.HELPER
