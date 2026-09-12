from __future__ import annotations

from pathlib import Path

from local_markup.studio_downloads import (
    build_engine_handoff_text,
    build_history_text,
    build_prompt_pack_text,
    history_gallery_markdown,
    write_history_download,
    write_prompt_pack,
)
from local_markup.studio_history import StudioHistoryItem, StudioHistoryStore


def test_build_prompt_pack_text_contains_copy_ready_sections() -> None:
    text = build_prompt_pack_text(
        workflow="Image Prompt",
        fooocus_area="Image Prompt tab",
        prompt="clean portrait",
        negative_prompt="blur",
        setup_steps="upload face reference",
        next_shots="Shot 2",
    )

    assert "Fooocus AI Studio Prompt Pack" in text
    assert "Workflow: Image Prompt" in text
    assert "Prompt:" in text
    assert "Negative Prompt:" in text
    assert "Setup Steps:" in text


def test_build_engine_handoff_text_reports_autofill_send_status() -> None:
    text = build_engine_handoff_text(
        workflow="Image Prompt",
        fooocus_area="Image Prompt tab",
        prompt="clean portrait",
        negative_prompt="blur",
        setup_steps="upload face reference",
        next_shots="Shot 2",
    )

    assert "Sent to Engine" in text
    assert "browser bridge" in text
    assert "Open the Hidden Fooocus engine" in text
    assert "click Generate" in text
    assert "Image Prompt tab" in text
    assert "clean portrait" in text
    assert "blur" in text


def test_build_engine_handoff_text_guides_user_to_build_plan_first() -> None:
    text = build_engine_handoff_text("", "", "", "", "", "")

    assert "Build your Fooocus plan first" in text


def test_write_prompt_pack_creates_download_file() -> None:
    path = Path(write_prompt_pack("workflow", "area", "prompt", "negative", "steps", "shots"))

    assert path.exists()
    assert "prompt" in path.read_text(encoding="utf-8")


def test_write_history_download_creates_download_file() -> None:
    path = Path(write_history_download("## History"))

    assert path.exists()
    assert "Fooocus AI Studio History" in path.read_text(encoding="utf-8")


def test_history_gallery_markdown_handles_empty_and_populated_store() -> None:
    empty = history_gallery_markdown(StudioHistoryStore())
    assert "No generated images" in empty

    populated = history_gallery_markdown(
        StudioHistoryStore(
            items=[StudioHistoryItem("item-1", "prompt", "negative", "text_to_image", image_path="image.png", rating=5)]
        )
    )
    assert "History Gallery" in populated
    assert "image.png" in populated
    assert "item-1" in populated


def test_build_history_text_wraps_markdown() -> None:
    assert build_history_text("session").startswith("Fooocus AI Studio History")
