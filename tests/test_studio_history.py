from __future__ import annotations

from local_markup.studio_history import StudioHistoryItem, StudioHistoryStore, create_history_item


def test_create_history_item_stores_prompt_workflow_and_metadata() -> None:
    item = create_history_item(
        item_id="item-1",
        prompt="clean product photo",
        negative_prompt="blur",
        workflow="text_to_image",
        image_path="outputs/item-1.png",
        seed=42,
        metadata={"source": "manual"},
    )

    assert item.item_id == "item-1"
    assert item.prompt == "clean product photo"
    assert item.negative_prompt == "blur"
    assert item.workflow == "text_to_image"
    assert item.image_path == "outputs/item-1.png"
    assert item.seed == 42
    assert item.metadata == {"source": "manual"}


def test_history_store_adds_latest_first() -> None:
    first = StudioHistoryItem("1", "first", "", "text_to_image")
    second = StudioHistoryItem("2", "second", "", "image_prompt")

    store = StudioHistoryStore().add(first).add(second)

    assert [item.item_id for item in store.latest()] == ["2", "1"]


def test_history_store_filters_by_workflow_and_favorites() -> None:
    store = StudioHistoryStore(
        items=[
            StudioHistoryItem("1", "a", "", "text_to_image", rating=5),
            StudioHistoryItem("2", "b", "", "image_prompt", rating=3),
            StudioHistoryItem("3", "c", "", "text_to_image", rating=4),
        ]
    )

    assert [item.item_id for item in store.by_workflow("text_to_image")] == ["1", "3"]
    assert [item.item_id for item in store.favorites()] == ["1", "3"]
    assert [item.item_id for item in store.latest(limit=2)] == ["1", "2"]
