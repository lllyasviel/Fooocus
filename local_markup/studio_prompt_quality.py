from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from local_markup.ai_studio_agent_v2 import AgentPlan


@dataclass(frozen=True)
class PromptQualityCriterion:
    key: str
    label: str
    pass_standard: str
    fail_signal: str


PROMPT_QUALITY_CRITERIA: Dict[str, PromptQualityCriterion] = {
    "specificity": PromptQualityCriterion(
        key="specificity",
        label="Specificity",
        pass_standard="Prompt describes one shot, one setting, one task, and one desired outcome.",
        fail_signal="Prompt is vague or combines too many outputs at once.",
    ),
    "feature_fit": PromptQualityCriterion(
        key="feature_fit",
        label="Feature fit",
        pass_standard="Prompt matches the selected Fooocus workflow.",
        fail_signal="Prompt does not match the selected workflow or control area.",
    ),
    "reference_handling": PromptQualityCriterion(
        key="reference_handling",
        label="Reference handling",
        pass_standard="Prompt explains how the source or reference should guide the output.",
        fail_signal="Prompt overpromises exact matching from limited source material.",
    ),
    "negative_prompt": PromptQualityCriterion(
        key="negative_prompt",
        label="Negative prompt",
        pass_standard="Negative prompt blocks common defects without fighting the desired output.",
        fail_signal="Negative prompt is too generic or contradicts the requested result.",
    ),
    "handoff_clarity": PromptQualityCriterion(
        key="handoff_clarity",
        label="Handoff clarity",
        pass_standard="User knows exactly which Fooocus tab or control to use.",
        fail_signal="User still has to guess the correct Fooocus workflow.",
    ),
}


@dataclass(frozen=True)
class PromptQualityReview:
    feature_key: str
    prompt: str
    negative_prompt: str
    handoff_recipe: str
    checklist: Dict[str, str]

    def as_markdown(self) -> str:
        lines: List[str] = [
            "## Prompt Quality Review",
            "",
            f"**Feature:** `{self.feature_key}`",
            "",
            "### Prompt",
            self.prompt,
            "",
            "### Negative prompt",
            self.negative_prompt,
            "",
            "### Handoff recipe",
            self.handoff_recipe,
            "",
            "### Manual scorecard",
        ]
        for key, status in self.checklist.items():
            criterion = PROMPT_QUALITY_CRITERIA[key]
            lines.extend(
                [
                    f"#### {criterion.label}",
                    f"- Status: {status}",
                    f"- Pass standard: {criterion.pass_standard}",
                    f"- Review cue: {criterion.fail_signal}",
                ]
            )
        return "\n".join(lines)


def build_prompt_quality_review(plan: AgentPlan) -> PromptQualityReview:
    """Create a manual prompt-quality review shell for a planner output.

    This does not claim automated image scoring. It creates a stable checklist
    reviewers can fill in after checking the generated plan.
    """

    return PromptQualityReview(
        feature_key=plan.feature.key,
        prompt=plan.primary_prompt,
        negative_prompt=plan.negative_prompt,
        handoff_recipe=plan.handoff_recipe,
        checklist={key: "manual-review-required" for key in PROMPT_QUALITY_CRITERIA},
    )
