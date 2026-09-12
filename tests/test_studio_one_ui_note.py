from __future__ import annotations

from local_markup.studio_one_ui_note import one_ui_note_markdown


def test_one_ui_note_mentions_single_work_surface_and_engine_url() -> None:
    markdown = one_ui_note_markdown()

    assert "One UI mode" in markdown
    assert "127.0.0.1:7872" in markdown
    assert "127.0.0.1:7865" in markdown
    assert "same page" in markdown
