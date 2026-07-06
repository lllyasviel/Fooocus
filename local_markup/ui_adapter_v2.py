from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .planner import build_edit_intent


ACTION_LABELS = {
    "remove": "Remove something",
    "recolor": "Change a color",
    "replace": "Replace part of the image",
    "enhance": "Improve the image",
}

MODE_LABELS = {
    "inpaint": "Edit the selected area",
    "enhance": "Improve the image",
}


@dataclass
class MarkupUiState:
    inpaint_prompt: str
    negative_prompt: str
    mode: str
    detection_prompt: str
    summary: str
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def target_to_detection_prompt(targets: List[str]) -> str:
    cleaned = [target for target in targets if target and target != "selected area"]
    if not cleaned:
        return "selected area"
    if "people" in cleaned:
        return "person"
    return cleaned[0]


def build_markup_ui_state(user_instruction: str, image_context: Optional[str] = None) -> MarkupUiState:
    intent = build_edit_intent(user_instruction, image_context=image_context)
    target_text = ", ".join(intent.targets) if intent.targets else "selected area"
    action_label = ACTION_LABELS.get(intent.action, intent.action)
    mode_label = MODE_LABELS.get(intent.mode, intent.mode)
    summary = (
        "Edit plan:\n"
        f"- Change: {action_label}\n"
        f"- Area to change: {target_text}\n"
        f"- Best tool: {mode_label}\n"
        "- The edit prompt has been filled in. Draw or upload a mask around the area first."
    )
    return MarkupUiState(
        inpaint_prompt=intent.prompt,
        negative_prompt=intent.negative_prompt,
        mode=intent.mode,
        detection_prompt=target_to_detection_prompt(intent.targets),
        summary=summary,
        notes=intent.notes,
    )


def build_markup_ui_outputs(user_instruction: str, image_context: Optional[str] = None):
    state = build_markup_ui_state(user_instruction, image_context=image_context)
    return state.inpaint_prompt, state.detection_prompt, state.summary, state.negative_prompt
