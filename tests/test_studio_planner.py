from __future__ import annotations

import pytest

from local_markup.ai_studio_agent_v2 import build_agent_plan, build_agent_outputs
from local_markup.studio_knowledge import FEATURES, REQUIRED_FEATURE_KEYS, route_feature, validate_feature_map


@pytest.mark.parametrize(
    ("goal", "expected_key"),
    [
        ("Create a clean product photo of a sneaker", "text_to_image"),
        ("Use this uploaded image as the visual reference", "image_prompt"),
        ("Make a standing lifestyle photo from this source", "face_reference"),
        ("Remove the jacket from this photo", "inpaint"),
        ("Use this pose but make a new image", "pyracanny"),
        ("Follow this composition but not every edge", "cpds"),
        ("Make this bigger and sharper", "upscale"),
        ("Make a similar but different version", "variation"),
        ("Describe this image as a prompt", "describe"),
    ],
)
def test_route_feature_common_scenarios(goal: str, expected_key: str) -> None:
    image_count = 1 if expected_key != "text_to_image" else 0
    wants_identity = expected_key == "face_reference"

    feature = route_feature(
        goal=goal,
        image_count=image_count,
        wants_identity=wants_identity,
        wants_exact_edit=False,
        wants_bundle=False,
    )

    assert feature.key == expected_key
    assert feature.fooocus_area


def test_route_feature_exact_edit_switch_wins() -> None:
    feature = route_feature(
        goal="Make this a cleaner image",
        image_count=1,
        wants_identity=False,
        wants_exact_edit=True,
        wants_bundle=False,
    )

    assert feature.key == "inpaint"


def test_route_feature_bundle_switch_wins() -> None:
    feature = route_feature(
        goal="Create a photo set",
        image_count=1,
        wants_identity=False,
        wants_exact_edit=False,
        wants_bundle=True,
    )

    assert feature.key == "face_reference"


def test_feature_map_contains_required_features() -> None:
    validate_feature_map()

    assert REQUIRED_FEATURE_KEYS == set(FEATURES)


def test_agent_plan_has_copy_ready_outputs() -> None:
    plan = build_agent_plan(
        goal="Make a realistic full body resort lifestyle photo from this source",
        image_count=1,
        wants_identity=True,
        wants_exact_edit=False,
        wants_bundle=False,
    )

    assert plan.feature.key == "face_reference"
    assert plan.primary_prompt
    assert plan.negative_prompt
    assert plan.shot_prompts
    assert plan.handoff_recipe
    assert plan.tool
    assert plan.fooocus_tab
    assert "Guardrails" not in plan.as_markdown()


def test_build_agent_outputs_shape() -> None:
    outputs = build_agent_outputs(
        goal="Use this pose but make a new image",
        image_1=object(),
        image_2=None,
        image_3=None,
        wants_identity=False,
        wants_exact_edit=False,
        wants_bundle=False,
    )

    assert len(outputs) == 7
    agent_plan, primary_prompt, negative_prompt, selected_tool, selected_area, shot_prompts, handoff_recipe = outputs
    assert "Agent Plan" in agent_plan
    assert primary_prompt
    assert negative_prompt
    assert selected_tool
    assert selected_area
    assert "Shot 1:" in shot_prompts
    assert handoff_recipe
