from __future__ import annotations


FOOOCUS_ENGINE_URL = "http://127.0.0.1:7865"
AI_STUDIO_URL = "http://127.0.0.1:7872"


def one_ui_note_markdown() -> str:
    return (
        "## One UI mode\n\n"
        f"Work from AI Studio at `{AI_STUDIO_URL}`. "
        f"The Fooocus engine is embedded below from `{FOOOCUS_ENGINE_URL}` so you do not have to switch browser tabs.\n\n"
        "Start the one-command launcher, then use the Studio plan and embedded Fooocus panel in the same page."
    )
