from __future__ import annotations

from local_markup.engine_queue_contract import (
    EngineJobKind,
    EngineJobProgress,
    EngineJobRequest,
    EngineJobStatus,
    create_queued_record,
    transition_job,
)


def test_create_queued_record_sets_initial_state() -> None:
    request = EngineJobRequest(kind=EngineJobKind.TEXT_TO_IMAGE, prompt="a clean product photo")

    record = create_queued_record(request, job_id="job-1")

    assert record.job_id == "job-1"
    assert record.request == request
    assert record.status == EngineJobStatus.QUEUED
    assert record.outputs == []
    assert record.error is None


def test_transition_job_preserves_identity_and_request() -> None:
    request = EngineJobRequest(kind=EngineJobKind.IMAGE_PROMPT, prompt="use this reference", references=["ref-1.png"])
    record = create_queued_record(request, job_id="job-2")

    running = transition_job(record, EngineJobStatus.RUNNING)
    done = transition_job(running, EngineJobStatus.SUCCEEDED, outputs=["out.png"])

    assert running.job_id == record.job_id
    assert running.request == request
    assert done.job_id == record.job_id
    assert done.outputs == ["out.png"]
    assert done.error is None


def test_transition_job_can_record_failure() -> None:
    request = EngineJobRequest(kind=EngineJobKind.INPAINT, prompt="replace background")
    record = create_queued_record(request, job_id="job-3")

    failed = transition_job(record, EngineJobStatus.FAILED, error="failed during validation")

    assert failed.status == EngineJobStatus.FAILED
    assert failed.error == "failed during validation"


def test_progress_percent_is_clamped() -> None:
    assert EngineJobProgress("job", EngineJobStatus.RUNNING, "start", step=0, total_steps=0).percent == 0.0
    assert EngineJobProgress("job", EngineJobStatus.RUNNING, "half", step=5, total_steps=10).percent == 50.0
    assert EngineJobProgress("job", EngineJobStatus.RUNNING, "over", step=12, total_steps=10).percent == 100.0
