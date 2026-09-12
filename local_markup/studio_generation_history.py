from __future__ import annotations

from hashlib import sha256
from typing import Optional

from local_markup.studio_adapter_contract import AdapterResult, ImageStudioJob
from local_markup.studio_history import StudioHistoryItem, StudioHistoryStore


def manual_history_id(job: ImageStudioJob) -> str:
    raw = "|".join([job.goal, job.prompt, job.negative_prompt, job.kind.value])
    digest = sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"manual-{digest}"


def reference_metadata(job: ImageStudioJob) -> dict[str, str]:
    metadata: dict[str, str] = {"reference_count": str(len(job.references))}
    for index, reference in enumerate(job.references, start=1):
        prefix = f"reference_{index}"
        metadata[f"{prefix}_name"] = reference.name
        metadata[f"{prefix}_path"] = reference.path or ""
        metadata[f"{prefix}_role"] = reference.role
    return metadata


def history_item_from_adapter_result(
    job: ImageStudioJob,
    result: AdapterResult,
    image_path: Optional[str] = None,
    rating: Optional[int] = None,
    notes: str = "",
) -> StudioHistoryItem:
    item_id = result.job_id or manual_history_id(job)
    metadata = dict(job.metadata)
    metadata.update(reference_metadata(job))
    metadata.update(
        {
            "adapter_status": result.status.value,
            "adapter_message": result.message,
        }
    )
    return StudioHistoryItem(
        item_id=item_id,
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        workflow=job.kind.value,
        image_path=image_path,
        seed=job.seed,
        rating=rating,
        notes=notes,
        metadata=metadata,
    )


def add_adapter_result_to_history(
    store: StudioHistoryStore,
    job: ImageStudioJob,
    result: AdapterResult,
    image_path: Optional[str] = None,
    rating: Optional[int] = None,
    notes: str = "",
) -> StudioHistoryStore:
    return store.add(
        history_item_from_adapter_result(
            job=job,
            result=result,
            image_path=image_path,
            rating=rating,
            notes=notes,
        )
    )
