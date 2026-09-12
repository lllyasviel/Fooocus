from __future__ import annotations

import pytest

from local_markup.ai_studio_agent_v2 import build_agent_plan
from local_markup.studio_prompt_quality import PROMPT_QUALITY_CRITERIA, build_prompt_quality_review


@pytest.mark.parametrize(
    ("goal", "image_count", "wants_identity", "wants_exact_edit", "wants_bundle"),
    [
        ("Create a clean product photo of a sneaker", 0, False, False, False),
        ("Use this uploaded image as a visual reference", 1, False, False, False),
        ("Make a standing lifestyle photo from this source", 1, True, False, False),
        ("Remove the jacket from this image", 1, False, True, False),
        ("Make this bigger and sharper", 1, False, False, False),
    ],
)
def test_prompt_quality_review_created_for_sample_plans(
    goal: str,
    image_count: int,
    wants_identity: bool,
    wants_exact_edit: bool,
    wants_bundle: bool,
) -> None:
    plan = build_agent_plan(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)

    review = build_prompt_quality_review(plan)

    assert review.feature_key == plan.feature.key
    assert review.prompt == plan.primary_prompt
    assert review.negative_prompt == plan.negative_prompt
    assert review.handoff_recipe == plan.handoff_recipe
    assert set(review.checklist) == set(PROMPT_QUALITY_CRITERIA)
    assert "manual-review-required" in set(review.checklist.values())


def test_prompt_quality_review_markdown_has_all_criteria() -> None:
    plan = build_agent_plan("Create a clean product photo of a sneaker", 0, False, False, False)
    review = build_prompt_quality_review(plan)
    markdown = review.as_markdown()

    assert "Prompt Quality Review" in markdown
    for criterion in PROMPT_QUALITY_CRITERIA.values():
        assert criterion.label in markdown
        assert criterion.pass_standard in markdown
