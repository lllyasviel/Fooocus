from local_markup.gradio_helpers import build_markup_ui_outputs


def test_build_markup_ui_outputs_contains_plan():
    prompt, plan = build_markup_ui_outputs("remove the trash from the table")
    assert "trash" in prompt
    assert "Action:" in plan
    assert "remove" in plan
