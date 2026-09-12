from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind, EngineJobStatus
from local_markup.local_fooocus_adapter import LocalDryRunFooocusAdapter
from local_markup.studio_adapter_contract import AdapterJobStatus, ImageStudioJob, ReferenceImage
from local_markup.studio_provider_registry import get_provider_adapter


def test_local_dry_run_prepares_queued_record_without_generation() -> None:
    adapter = LocalDryRunFooocusAdapter()
    job = ImageStudioJob(
        goal="make a clean reference-based image",
        prompt="clean studio photo",
        negative_prompt="blur",
        kind=EngineJobKind.IMAGE_PROMPT,
        references=[ReferenceImage(name="source", path="source.png")],
    )

    submission = adapter.prepare_submission(job)

    assert submission.record.status == EngineJobStatus.QUEUED
    assert submission.record.request.kind == EngineJobKind.IMAGE_PROMPT
    assert submission.record.request.references == ["source.png"]
    assert any("No active Fooocus worker call" in note for note in submission.validation_notes)


def test_local_dry_run_submit_returns_adapter_result() -> None:
    adapter = LocalDryRunFooocusAdapter()
    job = ImageStudioJob(
        goal="make image",
        prompt="clean image",
        negative_prompt="",
        kind=EngineJobKind.TEXT_TO_IMAGE,
    )

    result = adapter.submit(job)

    assert result.status == AdapterJobStatus.ACCEPTED
    assert result.job_id is not None
    assert "Active generation was not started" in result.message


def test_provider_registry_can_create_local_dry_run_adapter() -> None:
    adapter = get_provider_adapter("local_dry_run")

    assert isinstance(adapter, LocalDryRunFooocusAdapter)
