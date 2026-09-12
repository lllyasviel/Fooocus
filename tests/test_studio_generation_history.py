from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_contract import AdapterJobStatus, AdapterResult, ImageStudioJob, ReferenceImage
from local_markup.studio_generation_history import add_adapter_result_to_history, history_item_from_adapter_result, reference_metadata
from local_markup.studio_history import StudioHistoryStore


def test_history_item_from_adapter_result_preserves_job_fields() -> None:
    job = ImageStudioJob(
        goal="create image",
        prompt="clean product photo",
        negative_prompt="blur",
        kind=EngineJobKind.TEXT_TO_IMAGE,
        seed=99,
        references=[ReferenceImage(name="ref", path="ref.png", role="image_prompt")],
        metadata={"fooocus_area": "Text to Image"},
    )
    result = AdapterResult(status=AdapterJobStatus.ACCEPTED, message="accepted", job_id="job-1")

    item = history_item_from_adapter_result(job, result, image_path="out.png", rating=5, notes="winner")

    assert item.item_id == "job-1"
    assert item.prompt == "clean product photo"
    assert item.negative_prompt == "blur"
    assert item.workflow == "text_to_image"
    assert item.image_path == "out.png"
    assert item.seed == 99
    assert item.rating == 5
    assert item.notes == "winner"
    assert item.metadata["adapter_status"] == "accepted"
    assert item.metadata["reference_count"] == "1"
    assert item.metadata["reference_1_name"] == "ref"
    assert item.metadata["reference_1_path"] == "ref.png"
    assert item.metadata["reference_1_role"] == "image_prompt"


def test_reference_metadata_preserves_multiple_reference_roles() -> None:
    job = ImageStudioJob(
        goal="edit image",
        prompt="replace background",
        negative_prompt="changed person",
        kind=EngineJobKind.INPAINT,
        references=[
            ReferenceImage(name="source", path="source.png", role="inpaint_source"),
            ReferenceImage(name="mask", path="mask.png", role="inpaint_mask"),
        ],
    )

    metadata = reference_metadata(job)

    assert metadata["reference_count"] == "2"
    assert metadata["reference_1_role"] == "inpaint_source"
    assert metadata["reference_2_role"] == "inpaint_mask"
    assert metadata["reference_2_path"] == "mask.png"


def test_add_adapter_result_to_history_adds_latest_first() -> None:
    job = ImageStudioJob(
        goal="create image",
        prompt="clean image",
        negative_prompt="",
        kind=EngineJobKind.TEXT_TO_IMAGE,
    )
    result = AdapterResult(status=AdapterJobStatus.MANUAL_HANDOFF, message="handoff")

    store = add_adapter_result_to_history(StudioHistoryStore(), job, result)

    assert len(store.items) == 1
    assert store.items[0].prompt == "clean image"
    assert store.items[0].metadata["adapter_status"] == "manual_handoff"
