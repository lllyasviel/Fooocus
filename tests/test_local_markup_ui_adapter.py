from local_markup.ui_adapter import build_markup_ui_outputs, build_markup_ui_state


def test_build_markup_ui_state_for_recolor():
    state = build_markup_ui_state("make the shirt black")
    assert state.mode == "inpaint"
    assert state.detection_prompt == "shirt"
    assert "black" in state.inpaint_prompt
    assert "Action: recolor" in state.summary


def test_build_markup_ui_state_for_selected_area_default():
    state = build_markup_ui_state("make this look cleaner")
    assert state.mode == "enhance"
    assert state.detection_prompt == "selected area"
    assert "selected area" in state.summary


def test_build_markup_ui_outputs_order():
    prompt, detection, summary, negative = build_markup_ui_outputs("remove the trash on the table")
    assert "trash" in prompt
    assert detection in {"table", "trash"}
    assert "remove" in summary
    assert "artifacts" in negative
