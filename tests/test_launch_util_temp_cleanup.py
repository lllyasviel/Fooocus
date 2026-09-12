from __future__ import annotations

from pathlib import Path

from modules.launch_util import delete_folder_content


def test_delete_folder_content_creates_missing_folder(tmp_path: Path) -> None:
    missing = tmp_path / "fooocus-temp"

    assert delete_folder_content(str(missing), "[Cleanup] ") is True
    assert missing.exists()
    assert missing.is_dir()


def test_delete_folder_content_removes_existing_content(tmp_path: Path) -> None:
    folder = tmp_path / "fooocus-temp"
    folder.mkdir()
    file_path = folder / "old.txt"
    file_path.write_text("old", encoding="utf-8")

    assert delete_folder_content(str(folder), "[Cleanup] ") is True
    assert list(folder.iterdir()) == []
