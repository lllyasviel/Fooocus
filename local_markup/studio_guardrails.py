from __future__ import annotations

from dataclasses import dataclass
from typing import List

from local_markup.studio_knowledge import StudioFeature


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    mode: str
    feature_key: str
    risk_level: str
    warnings: List[str]
    blocked_reason: str = ""

    def as_text(self) -> str:
        status = "Allowed" if self.allowed else "Blocked"
        lines = [
            f"Status: {status}",
            f"Mode: {self.mode}",
            f"Feature: {self.feature_key}",
            f"Risk level: {self.risk_level}",
        ]
        if self.blocked_reason:
            lines.append(f"Reason: {self.blocked_reason}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend([f"- {warning}" for warning in self.warnings])
        return "\n".join(lines)


def guardrail_mode() -> str:
    return "built_in"


def validate_feature_use(feature: StudioFeature, goal: str, image_count: int, wants_identity: bool) -> GuardrailDecision:
    mode = guardrail_mode()
    warnings: List[str] = []

    if feature.key in {"face_reference", "image_prompt"} and wants_identity:
        warnings.append("Identity-sensitive workflows should use user-provided or authorized images only.")
        if image_count == 0:
            warnings.append("No reference image detected. Subject consistency will be weak or impossible.")
        elif image_count == 1:
            warnings.append("Only one reference image detected. Full-length or bundle outputs may drift; add upper-body/full-length references when possible.")

    if feature.key == "inpaint":
        warnings.append("Exact edits require mask review before generation. Do not trust an automatic mask blindly.")

    if feature.key == "auto_mask_sam":
        warnings.append("SAM/GroundingDINO masks can select the wrong region. Review the mask before generation.")

    if feature.key == "auto_mask_u2net":
        warnings.append("Broad segmentation masks can include too much. Review the mask before generation.")

    if feature.key == "variation":
        warnings.append("Variation creates a new version inspired by the image; it is not an exact edit and may drift subject consistency.")

    if feature.key == "describe":
        warnings.append("Review image-to-prompt descriptions before applying them to a personal image workflow.")

    if feature.risk_level == "high":
        warnings.append("High-risk workflow: review references, prompts, and masks before generation.")

    return GuardrailDecision(True, mode, feature.key, feature.risk_level, warnings)
