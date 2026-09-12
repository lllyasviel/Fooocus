from __future__ import annotations

from ai_studio_app import build_app
from local_markup.studio_workflow_controller import build_studio_workflow_outputs


def test_build_studio_workflow_outputs_includes_preview_sections() -> None:
    outputs = build_studio_workflow_outputs(
        goal="Create a clean product photo",
        image_1=None,
        image_2=None,
        image_3=None,
        wants_identity=False,
        wants_exact_edit=False,
        wants_bundle=False,
        vram_gb=6,
    )

    assert len(outputs) == 9
    assert "## Ready for Fooocus" in outputs[7]
    assert "Generate one image first" in outputs[7]
    assert "## Local Session History" in outputs[8]


def test_ai_studio_app_builds_without_launching() -> None:
    app = build_app()

    assert app is not None
