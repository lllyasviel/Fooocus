from __future__ import annotations

from typing import Tuple

from .planner import build_edit_intent
from .presets import get_preset


def _format_plan(action: str, mode: str, targets: list[str], prompt: str, negative_prompt: str) -> str:
    target_text = ", ".join(targets)
    return (
        "### Edit Plan\n"
        f"- Action: `{action}`\n"
        f"- Mode: `{mode}`\n"
        f"- Target: {target_text}\n\n"
        "### Generated Inpaint Prompt\n"
        f"{prompt}\n\n"
        "### Suggested Negative Prompt\n"
        f"{negative_prompt}"
    )


def build_markup_ui_outputs(user_instruction: str) -> Tuple[str, str]:
    intent = build_edit_intent(user_instruction)
    plan = _format_plan(
        action=intent.action,
        mode=intent.mode,
        targets=intent.targets,
        prompt=intent.prompt,
        negative_prompt=intent.negative_prompt,
    )
    return intent.prompt, plan


def preset_to_ui_outputs(preset_name: str | None) -> Tuple[str, str, str]:
    if not preset_name:
        return "", "", "Choose a preset first."

    preset = get_preset(preset_name)
    instruction = preset["instruction"]
    prompt, plan = build_markup_ui_outputs(instruction)
    return instruction, prompt, plan
