from __future__ import annotations

from dataclasses import dataclass
from typing import List

from local_markup.studio_adapter_contract import ImageStudioJob, ManualHandoffAdapter
from local_markup.studio_adapter_preview import adapter_job_markdown


@dataclass(frozen=True)
class ManualSubmitPackage:
    provider: str
    workflow: str
    prompt: str
    negative_prompt: str
    references: List[str]
    metadata: dict[str, str]
    preview_markdown: str
    handoff_steps: List[str]


def build_manual_submit_package(job: ImageStudioJob) -> ManualSubmitPackage:
    adapter = ManualHandoffAdapter()
    result = adapter.submit(job)
    return ManualSubmitPackage(
        provider=adapter.name,
        workflow=job.kind.value,
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        references=[reference.path or reference.name for reference in job.references],
        metadata=dict(job.metadata),
        preview_markdown=adapter_job_markdown(job),
        handoff_steps=result.handoff_steps,
    )
