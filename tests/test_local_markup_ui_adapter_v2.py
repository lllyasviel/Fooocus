from local_markup import build_markup_ui_outputs, build_markup_ui_state


def test_ui_state_for_recolor():
    state = build_markup_ui_state("make the shirt black")
    assert state.mode == "inpaint"
    assert state.detection_prompt == "shirt"
    assert "black" in state.inpaint_prompt
    assert "Change a color" in state.summary


def test_ui_state_default_target():
    state = build_markup_ui_state("make this look cleaner")
    assert state.mode == "enhance"
    assert state.detection_prompt == "selected area"


def test_ui_outputs_order():
    prompt, detection, summary, negative = build_markup_ui_outputs("remove the clutter on the table")
    assert "clutter" in prompt
    assert detection in {"clutter", "table"}
    assert "Remove something" in summary
    assert "artifacts" in negative
