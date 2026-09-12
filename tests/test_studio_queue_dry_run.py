from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_contract import AdapterJobStatus, ImageStudioJob, ReferenceImage
from local_markup.studio_queue_dry_run import submit_dry_run_and_record_history


def test_submit_dry_run_and_record_history_adds_history_item() -> None:
    job = ImageStudioJob(
        goal="test dry run",
        prompt="clean portrait",
        negative_prompt="blur",
        kind=EngineJobKind.IMAGE_PROMPT,
        references=[ReferenceImage(name="face", path="face.png", role="face_reference")],
    )

    result = submit_dry_run_and_record_history(job)

    assert result.adapter_result.status == AdapterJobStatus.ACCEPTED
    assert result.adapter_result.job_id is not None
    assert len(result.history.items) == 1
    assert result.history.items[0].item_id == result.adapter_result.job_id
    assert result.history.items[0].metadata["reference_1_role"] == "face_reference"
    assert "No active generation" in result.history.items[0].notes
