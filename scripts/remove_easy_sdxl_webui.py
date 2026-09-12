from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui.py"

MARKERS = [
    ("# EASY_SDXL_WEBUI_HELPER_START", "# EASY_SDXL_WEBUI_HELPER_END"),
    ("# EASY_SDXL_WEBUI_PANEL_START", "# EASY_SDXL_WEBUI_PANEL_END"),
    ("# EASY_SDXL_WEBUI_BINDING_START", "# EASY_SDXL_WEBUI_BINDING_END"),
]


def remove_marked_block(text: str, start: str, end: str) -> tuple[str, bool]:
    start_index = text.find(start)
    if start_index == -1:
        return text, False

    block_start = text.rfind("\n", 0, start_index)
    if block_start == -1:
        block_start = start_index

    block_end = text.find(end, start_index)
    if block_end == -1:
        return text, False

    block_end = text.find("\n", block_end)
    if block_end == -1:
        block_end = len(text)

    return text[:block_start] + text[block_end:], True


def main():
    text = WEBUI.read_text(encoding="utf-8")
    changed = False

    for start, end in MARKERS:
        while start in text:
            text, removed = remove_marked_block(text, start, end)
            changed = changed or removed
            if not removed:
                break

    if changed:
        WEBUI.write_text(text, encoding="utf-8")
        print("Removed old Easy SDXL WebUI patch blocks from webui.py.")
    else:
        print("No old Easy SDXL WebUI patch blocks found in webui.py.")


if __name__ == "__main__":
    main()
