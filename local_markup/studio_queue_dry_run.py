from __future__ import annotations

from dataclasses import dataclass

from local_markup.local_fooocus_adapter import LocalDryRunFooocusAdapter
from local_markup.studio_adapter_contract import AdapterResult, ImageStudioJob
from local_markup.studio_generation_history import add_adapter_result_to_history
from local_markup.studio_history import StudioHistoryStore


@dataclass(frozen=True)
class StudioDryRunResult:
    adapter_result: AdapterResult
    history: StudioHistoryStore


def submit_dry_run_and_record_history(job: ImageStudioJob, history: StudioHistoryStore | None = None) -> StudioDryRunResult:
    adapter = LocalDryRunFooocusAdapter()
    adapter_result = adapter.submit(job)
    next_history = add_adapter_result_to_history(
        store=history or StudioHistoryStore(),
        job=job,
        result=adapter_result,
        notes="Dry-run submission recorded. No active generation was started.",
    )
    return StudioDryRunResult(adapter_result=adapter_result, history=next_history)
