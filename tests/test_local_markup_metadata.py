import json

from local_markup.metadata import MarkupSession, save_markup_session


def test_save_markup_session(tmp_path):
    session = MarkupSession(
        source_image="input.png",
        instruction="remove trash",
        prompt="clean realistic image",
        negative_prompt="artifacts",
        mode="inpaint",
        mask_image="mask.png",
    )
    saved = save_markup_session(session, str(tmp_path))
    assert saved.exists()
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["source_image"] == "input.png"
    assert data["mode"] == "inpaint"
    assert data["instruction"] == "remove trash"
