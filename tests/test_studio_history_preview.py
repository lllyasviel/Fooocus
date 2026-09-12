from __future__ import annotations

from local_markup.studio_history import StudioHistoryItem, StudioHistoryStore
from local_markup.studio_history_preview import history_store_markdown


def test_history_store_markdown_handles_empty_store() -> None:
    markdown = history_store_markdown(StudioHistoryStore())

    assert "No history items yet" in markdown


def test_history_store_markdown_lists_latest_items() -> None:
    store = StudioHistoryStore(
        items=[
            StudioHistoryItem("1", "first prompt", "", "text_to_image", image_path="one.png", rating=5),
            StudioHistoryItem("2", "second prompt", "bad", "image_prompt"),
        ]
    )

    markdown = history_store_markdown(store, limit=1)

    assert "Studio History" in markdown
    assert "first prompt" in markdown
    assert "one.png" in markdown
    assert "second prompt" not in markdown
