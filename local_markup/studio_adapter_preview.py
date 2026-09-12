from __future__ import annotations

from local_markup.studio_adapter_mappings import AdapterMappingPlan
from local_markup.studio_adapter_contract import ImageStudioJob


def adapter_job_markdown(job: ImageStudioJob) -> str:
    references = job.references or []
    reference_lines = [f"- `{item.role}`: {item.path or item.name}" for item in references]
    if not reference_lines:
        reference_lines = ["- None"]

    metadata_lines = [f"- `{key}`: {value}" for key, value in sorted(job.metadata.items())]
    if not metadata_lines:
        metadata_lines = ["- None"]

    return "\n".join(
        [
            f"## Adapter Job Preview: {job.kind.value}",
            "",
            f"Goal: {job.goal}",
            "",
            "### Prompt",
            job.prompt,
            "",
            "### Negative Prompt",
            job.negative_prompt or "None",
            "",
            "### References",
            *reference_lines,
            "",
            "### Metadata",
            *metadata_lines,
        ]
    )


def adapter_mapping_markdown(plan: AdapterMappingPlan) -> str:
    note_lines = [f"- `{note.field}` = `{note.value}` — {note.note}" for note in plan.notes]
    return "\n".join([adapter_job_markdown(plan.job), "", "### Mapping Notes", *note_lines])
