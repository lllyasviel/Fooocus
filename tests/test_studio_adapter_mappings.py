from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_mappings import build_face_reference_job, build_image_prompt_job, build_inpaint_job


def test_build_image_prompt_job_preserves_reference_order() -> None:
    plan = build_image_prompt_job(
        goal="use these references",
        prompt="clean editorial image",
        negative_prompt="blur",
        reference_paths=["a.png", "b.png"],
    )

    assert plan.job.kind == EngineJobKind.IMAGE_PROMPT
    assert [ref.path for ref in plan.job.references] == ["a.png", "b.png"]
    assert [ref.role for ref in plan.job.references] == ["image_prompt", "image_prompt"]
    assert plan.job.metadata["fooocus_area"] == "Image Prompt"


def test_build_face_reference_job_marks_reference_mode() -> None:
    plan = build_face_reference_job(
        goal="keep this person recognizable",
        prompt="realistic lifestyle photo",
        negative_prompt="distorted face",
        face_reference_path="face.png",
    )

    assert plan.job.kind == EngineJobKind.IMAGE_PROMPT
    assert plan.job.references[0].path == "face.png"
    assert plan.job.references[0].role == "face_reference"
    assert plan.job.metadata["reference_mode"] == "face_reference"
    assert any(note.field == "reference_mode" for note in plan.notes)


def test_build_inpaint_job_maps_source_and_mask() -> None:
    plan = build_inpaint_job(
        goal="replace only the background",
        prompt="modern office background",
        negative_prompt="changed person",
        source_image_path="source.png",
        mask_path="mask.png",
    )

    assert plan.job.kind == EngineJobKind.INPAINT
    assert [ref.role for ref in plan.job.references] == ["inpaint_source", "inpaint_mask"]
    assert [ref.path for ref in plan.job.references] == ["source.png", "mask.png"]
    assert plan.job.metadata["fooocus_area"] == "Inpaint"
    assert plan.job.metadata["edit_mode"] == "masked_edit"
    assert any(note.field == "mask" for note in plan.notes)
