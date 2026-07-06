import pytest

from local_markup.presets import get_preset


def test_get_known_preset():
    preset = get_preset("remove_object")
    assert preset["mode"] == "inpaint"
    assert "remove" in preset["instruction"]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        get_preset("missing")
