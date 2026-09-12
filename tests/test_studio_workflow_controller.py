from __future__ import annotations

from local_markup.engine_queue_contract import EngineJobKind
from local_markup.studio_adapter_contract import AdapterJobStatus
from local_markup.studio_workflow_controller import (
    build_studio_workflow_outputs,
    reference_names_from_count,
    run_studio_workflow,
)


def test_reference_names_from_count_are_stable() -> None:
    assert reference_names_from_count(0) == []
    assert reference_names_from_count(3) == ["studio_reference_1", "studio_reference_2", "studio_reference_3"]


def test_run_studio_workflow_creates_plan_adapter_and_history() -> None:
    run = run_studio_workflow(
        goal="Make a realistic full body resort lifestyle photo from this source",
        image_count=1,
        wants_identity=True,
        wants_exact_edit=False,
        wants_bundle=False,
        vram_gb=6,
    )

    assert run.plan.feature.key == "face_reference"
    assert run.job.kind == EngineJobKind.IMAGE_PROMPT
    assert run.job.references[0].role == "face_reference"
    assert run.adapter_result.status == AdapterJobStatus.ACCEPTED
    assert run.adapter_result.job_id is not None
    assert run.hardware_profile.key == "low_vram_6gb"
    assert len(run.history.items) == 1
    assert run.history.items[0].metadata["adapter_status"] == "accepted"
    assert run.history.items[0].metadata["reference_count"] == "1"


def test_build_job_from_plan_maps_inpaint_to_source_and_mask() -> None:
    run = run_studio_workflow(
        goal="Replace the background only with a modern office",
        image_count=2,
        wants_identity=False,
        wants_exact_edit=True,
        wants_bundle=False,
        vram_gb=6,
    )

    assert run.plan.feature.key == "inpaint"
    assert run.job.kind == EngineJobKind.INPAINT
    assert [ref.role for ref in run.job.references] == ["inpaint_source", "inpaint_mask"]
    assert run.job.metadata["fooocus_area"] == "Inpaint"


def test_build_studio_workflow_outputs_shape_matches_ui_outputs() -> None:
    outputs = build_studio_workflow_outputs(
        "Create a polished product image",
        None,
        None,
        None,
        True,
        False,
        False,
        6,
    )

    assert len(outputs) == 9
    assert "## Agent Plan" in outputs[0]
    assert outputs[1]
    assert outputs[2]
    assert outputs[3]
    assert outputs[4]
    assert "Shot 1:" in outputs[5]
    assert outputs[6]
    assert "## Ready for Fooocus" in outputs[7]
    assert "Simple setup steps" in outputs[7]
    assert "## Local Session History" in outputs[8]
