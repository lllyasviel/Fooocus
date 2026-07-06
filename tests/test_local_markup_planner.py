import pytest

from local_markup import build_edit_intent
from local_markup.planner import detect_action, detect_colors, detect_targets


def test_detect_remove_action():
    assert detect_action("remove the trash on the table") == "remove"


def test_detect_recolor_action_and_color():
    intent = build_edit_intent("make the shirt black")
    assert intent.action == "recolor"
    assert intent.mode == "inpaint"
    assert "shirt" in intent.targets
    assert "black" in intent.prompt


def test_detect_replace_background():
    intent = build_edit_intent("replace the background with a luxury apartment")
    assert intent.action == "replace"
    assert intent.mode == "inpaint"
    assert "background" in intent.targets


def test_detect_enhance_default():
    intent = build_edit_intent("make this look more professional")
    assert intent.action == "enhance"
    assert intent.mode == "enhance"
    assert intent.targets == ["selected area"]


def test_empty_instruction_raises():
    with pytest.raises(ValueError):
        build_edit_intent("   ")


def test_detect_colors_multiple():
    assert detect_colors("make the car black with gold trim") == ["black", "gold"]


def test_detect_targets_unique_sorted():
    assert detect_targets("remove the trash and clutter from the table") == ["clutter", "table", "trash"]
