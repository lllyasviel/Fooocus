from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .planner import build_edit_intent


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


def _target_to_detection_prompt(targets: List[str]) -> str:
    cleaned = [target for target in targets if target and target != "selected area"]
    if not cleaned:
        return "selected area"
    if "people" in cleaned:
        return "person"
    return cleaned[0]


def _summary(action: str, targets: List[str], mode: str) -> str:
    target_text = ", ".join(targets) if targets else "selected area"
    return f"Action: {action} | Target: {target_text} | Suggested mode: {mode}"


def build_markup_ui_state(user_instruction: str, image_context: Optional[str] = None) -> MarkupUiState:
    intent = build_edit_intent(user_instruction, image_context=image_context)
    detection_prompt = _target_to_detection_prompt(intent.targets)
    return MarkupUiState(
        inpaint_prompt=intent.prompt,
        negative_prompt=intent.negative_prompt,
        mode=intent.mode,
        detection_prompt=detection_prompt,
        summary=_summary(intent.action, intent.targets, intent.mode),
        notes=intent.notes,
    )


def build_markup_ui_outputs(user_instruction: str, image_context: Optional[str] = None):
    """Return tuple-friendly outputs for Gradio callbacks.

    Intended output order:
    1. inpaint additional prompt
    2. detection prompt
    3. summary text
    4. negative prompt suggestion
    """
    state = build_markup_ui_state(user_instruction, image_context=image_context)
    return state.inpaint_prompt, state.detection_prompt, state.summary, state.negative_prompt
