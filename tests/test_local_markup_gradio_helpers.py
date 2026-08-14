from local_markup.gradio_helpers import build_markup_ui_outputs, preset_to_ui_outputs


def test_build_markup_ui_outputs_contains_plan():
    prompt, plan = build_markup_ui_outputs("remove the trash from the table")
    assert "trash" in prompt
    assert "Edit Plan" in plan
    assert "remove" in plan


def test_preset_to_ui_outputs():
    instruction, prompt, plan = preset_to_ui_outputs("clean_background")
    assert instruction
    assert prompt
    assert "Edit Plan" in plan


def test_empty_preset_returns_message():
    instruction, prompt, plan = preset_to_ui_outputs(None)
    assert instruction == ""
    assert prompt == ""
    assert "preset" in plan.lower()
