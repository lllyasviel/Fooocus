from __future__ import annotations

from dataclasses import dataclass
from typing import List

from local_markup.engine_queue_contract import EngineJobRecord, create_queued_record
from local_markup.studio_adapter_contract import AdapterJobStatus, AdapterResult, ImageStudioJob


@dataclass(frozen=True)
class LocalDryRunSubmission:
    record: EngineJobRecord
    validation_notes: List[str]


class LocalDryRunFooocusAdapter:
    name = "local_dry_run"

    def submit(self, job: ImageStudioJob) -> AdapterResult:
        submission = self.prepare_submission(job)
        return AdapterResult(
            status=AdapterJobStatus.ACCEPTED,
            message="Dry-run local Fooocus job accepted. Active generation was not started.",
            job_id=submission.record.job_id,
            handoff_steps=submission.validation_notes,
        )

    def prepare_submission(self, job: ImageStudioJob) -> LocalDryRunSubmission:
        request = job.to_engine_request()
        record = create_queued_record(request)
        notes = [
            f"Prepared job kind: {request.kind.value}",
            f"Prompt length: {len(request.prompt)}",
            f"Reference count: {len(request.references)}",
            "No active Fooocus worker call was made.",
        ]
        return LocalDryRunSubmission(record=record, validation_notes=notes)
