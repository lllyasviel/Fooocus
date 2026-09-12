from __future__ import annotations

from local_markup.studio_control_ui import CONTROL_UI_CSS, engine_hidden_note, history_gallery_empty_note, studio_hero_markdown


def test_studio_control_ui_has_designed_shell_copy() -> None:
    hero = studio_hero_markdown()

    assert "Control Center" in hero
    assert "hidden Fooocus engine" in hero
    assert "one first shot" in hero
    assert "studio-hero" in CONTROL_UI_CSS


def test_engine_hidden_and_history_notes_are_clear() -> None:
    assert "hidden by default" in engine_hidden_note()
    assert "prompt pack" in history_gallery_empty_note()
