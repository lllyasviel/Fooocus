from __future__ import annotations

from local_markup.studio_usage_guide import studio_usage_markdown


def test_studio_usage_markdown_has_fast_steps_and_no_friction_rules() -> None:
    markdown = studio_usage_markdown()

    assert "Fastest way to use this Studio" in markdown
    assert "Start Fooocus" in markdown
    assert "Open Fooocus" in markdown
    assert "No-friction rules" in markdown
    assert "one goal" in markdown
