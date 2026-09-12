from __future__ import annotations

from local_markup.studio_history import StudioHistoryItem, StudioHistoryStore


def history_item_markdown(item: StudioHistoryItem) -> str:
    image_line = item.image_path or "No image path recorded"
    rating_line = "unrated" if item.rating is None else str(item.rating)
    return "\n".join(
        [
            f"### {item.workflow} — {item.item_id}",
            f"Image: `{image_line}`",
            f"Rating: `{rating_line}`",
            "",
            "Prompt:",
            item.prompt,
            "",
            "Negative prompt:",
            item.negative_prompt or "None",
            "",
            f"Notes: {item.notes or 'None'}",
        ]
    )


def history_store_markdown(store: StudioHistoryStore, limit: int = 10) -> str:
    items = store.latest(limit=limit)
    if not items:
        return "## Studio History\n\nNo history items yet."

    sections = ["## Studio History", ""]
    for item in items:
        sections.append(history_item_markdown(item))
        sections.append("")
    return "\n".join(sections).rstrip()
