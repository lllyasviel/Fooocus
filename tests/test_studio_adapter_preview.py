from __future__ import annotations

from local_markup.studio_adapter_mappings import build_inpaint_job
from local_markup.studio_adapter_preview import adapter_mapping_markdown


def test_adapter_mapping_markdown_shows_references_and_notes() -> None:
    plan = build_inpaint_job(
        goal="replace the background",
        prompt="modern office background",
        negative_prompt="changed person",
        source_image_path="source.png",
        mask_path="mask.png",
    )

    markdown = adapter_mapping_markdown(plan)

    assert "Adapter Job Preview: inpaint" in markdown
    assert "`inpaint_source`: source.png" in markdown
    assert "`inpaint_mask`: mask.png" in markdown
    assert "Mapping Notes" in markdown
    assert "Fooocus area" not in markdown
    assert "fooocus_area" in markdown
