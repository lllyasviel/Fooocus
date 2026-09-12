from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_contract import ImageStudioJob, ReferenceImage
from local_markup.studio_manual_submit_package import build_manual_submit_package


def test_build_manual_submit_package_contains_copy_ready_fields() -> None:
    job = ImageStudioJob(
        goal="keep face and change outfit",
        prompt="realistic fashion portrait",
        negative_prompt="bad hands",
        kind=EngineJobKind.IMAGE_PROMPT,
        references=[ReferenceImage(name="face", path="face.png", role="face_reference")],
        metadata={"reference_mode": "face_reference"},
    )

    package = build_manual_submit_package(job)

    assert package.provider == "manual_handoff"
    assert package.workflow == "image_prompt"
    assert package.prompt == "realistic fashion portrait"
    assert package.negative_prompt == "bad hands"
    assert package.references == ["face.png"]
    assert package.metadata == {"reference_mode": "face_reference"}
    assert "Adapter Job Preview" in package.preview_markdown
    assert any("Copy the prompt" in step for step in package.handoff_steps)
