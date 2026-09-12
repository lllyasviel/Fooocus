from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_contract import (
    AdapterJobStatus,
    ImageStudioJob,
    ManualHandoffAdapter,
    ReferenceImage,
)


def test_image_studio_job_converts_to_engine_request() -> None:
    job = ImageStudioJob(
        goal="create a realistic portrait",
        prompt="realistic portrait in soft light",
        negative_prompt="blur, artifacts",
        kind=EngineJobKind.IMAGE_PROMPT,
        references=[ReferenceImage(name="face", path="face.png", role="subject")],
        width=1024,
        height=1024,
        seed=123,
        metadata={"source": "studio"},
    )

    request = job.to_engine_request()

    assert request.kind == EngineJobKind.IMAGE_PROMPT
    assert request.prompt == job.prompt
    assert request.negative_prompt == job.negative_prompt
    assert request.references == ["face.png"]
    assert request.width == 1024
    assert request.height == 1024
    assert request.seed == 123
    assert request.metadata == {"source": "studio"}


def test_manual_handoff_adapter_does_not_start_generation() -> None:
    adapter = ManualHandoffAdapter()
    job = ImageStudioJob(
        goal="create a clean product photo",
        prompt="clean product photo on white background",
        negative_prompt="blur",
        kind=EngineJobKind.TEXT_TO_IMAGE,
    )

    result = adapter.submit(job)

    assert result.status == AdapterJobStatus.MANUAL_HANDOFF
    assert result.job_id is None
    assert "No automatic generation" in result.message
    assert any("Copy the prompt" in step for step in result.handoff_steps)
