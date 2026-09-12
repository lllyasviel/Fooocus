from __future__ import annotations

import inspect

import ai_studio_app
from ai_studio_app import build_app, copyable_textbox, fooocus_iframe_html
from local_markup.studio_one_ui_note import FOOOCUS_ENGINE_URL, one_ui_note_markdown


def test_copyable_textbox_helper_enables_copy_button() -> None:
    source = inspect.getsource(copyable_textbox)

    assert "show_copy_button=True" in source


def test_ai_studio_app_uses_copy_controls_and_hidden_engine_panel() -> None:
    source = inspect.getsource(ai_studio_app.build_app)

    assert "Hidden Fooocus engine" in source
    assert "Use copy buttons" in source or "copy buttons" in source
    assert "copyable_textbox" in source


def test_fooocus_iframe_points_to_local_engine() -> None:
    html = fooocus_iframe_html()

    assert FOOOCUS_ENGINE_URL in html
    assert "iframe" in html
    assert "height:78vh" in html


def test_one_ui_note_is_clear() -> None:
    markdown = one_ui_note_markdown()

    assert "One UI mode" in markdown
    assert "same page" in markdown


def test_ai_studio_app_builds_one_ui_without_launching() -> None:
    app = build_app()

    assert app is not None
