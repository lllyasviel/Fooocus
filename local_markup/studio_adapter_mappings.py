from __future__ import annotations

from dataclasses import dataclass
from typing import List

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_contract import ImageStudioJob, ReferenceImage


@dataclass(frozen=True)
class AdapterMappingNote:
    field: str
    value: str
    note: str


@dataclass(frozen=True)
class AdapterMappingPlan:
    job: ImageStudioJob
    notes: List[AdapterMappingNote]


def build_image_prompt_job(goal: str, prompt: str, negative_prompt: str, reference_paths: List[str]) -> AdapterMappingPlan:
    references = [ReferenceImage(name=f"image_prompt_{index + 1}", path=path, role="image_prompt") for index, path in enumerate(reference_paths)]
    job = ImageStudioJob(
        goal=goal,
        prompt=prompt,
        negative_prompt=negative_prompt,
        kind=EngineJobKind.IMAGE_PROMPT,
        references=references,
        metadata={"fooocus_area": "Image Prompt"},
    )
    notes = [
        AdapterMappingNote("fooocus_area", "Image Prompt", "Use regular image prompt influence for visual guidance."),
        AdapterMappingNote("reference_count", str(len(references)), "All provided reference images are preserved in order."),
    ]
    return AdapterMappingPlan(job=job, notes=notes)


def build_face_reference_job(goal: str, prompt: str, negative_prompt: str, face_reference_path: str) -> AdapterMappingPlan:
    job = ImageStudioJob(
        goal=goal,
        prompt=prompt,
        negative_prompt=negative_prompt,
        kind=EngineJobKind.IMAGE_PROMPT,
        references=[ReferenceImage(name="face_reference", path=face_reference_path, role="face_reference")],
        metadata={"fooocus_area": "Image Prompt", "reference_mode": "face_reference"},
    )
    notes = [
        AdapterMappingNote("fooocus_area", "Image Prompt", "Use the Image Prompt tab with face-reference behavior."),
        AdapterMappingNote("reference_mode", "face_reference", "Keep identity guidance separate from general composition references."),
    ]
    return AdapterMappingPlan(job=job, notes=notes)


def build_inpaint_job(goal: str, prompt: str, negative_prompt: str, source_image_path: str, mask_path: str) -> AdapterMappingPlan:
    job = ImageStudioJob(
        goal=goal,
        prompt=prompt,
        negative_prompt=negative_prompt,
        kind=EngineJobKind.INPAINT,
        references=[
            ReferenceImage(name="source_image", path=source_image_path, role="inpaint_source"),
            ReferenceImage(name="mask", path=mask_path, role="inpaint_mask"),
        ],
        metadata={"fooocus_area": "Inpaint", "edit_mode": "masked_edit"},
    )
    notes = [
        AdapterMappingNote("fooocus_area", "Inpaint", "Use the Inpaint tab for local masked edits."),
        AdapterMappingNote("source_image", source_image_path, "Upload this as the image to edit."),
        AdapterMappingNote("mask", mask_path, "Use this mask to limit the edit area."),
    ]
    return AdapterMappingPlan(job=job, notes=notes)
