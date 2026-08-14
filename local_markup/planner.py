from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class EditIntent:
    user_instruction: str
    action: str
    targets: List[str]
    style: str
    mode: str
    prompt: str
    negative_prompt: str
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


COLOR_WORDS = {
    "red", "blue", "green", "black", "white", "gray", "grey", "yellow", "orange",
    "purple", "pink", "brown", "gold", "silver", "beige", "cream", "navy"
}

TARGET_WORDS = {
    "shirt", "face", "skin", "hair", "background", "sky", "car", "table", "floor",
    "wall", "room", "person", "people", "object", "trash", "clutter", "logo", "text"
}

ACTION_HINTS = {
    "remove": ["remove", "erase", "delete", "take out", "get rid"],
    "recolor": ["color", "recolor", "make it", "change to", "turn"],
    "replace": ["replace", "swap", "change background", "new background"],
    "enhance": ["improve", "clean", "polish", "better", "enhance", "fix", "sharp", "brighten"],
}

PRESET_PROMPTS = {
    "remove": "clean realistic image, natural fill, consistent background, no visible edit seams",
    "recolor": "realistic material color change, natural shadows, consistent texture, clean edges",
    "replace": "realistic scene replacement, natural lighting, matching perspective, believable depth",
    "enhance": "clean realistic enhancement, improved lighting, sharp details, natural colors",
}

NEGATIVE_BASE = "artifacts, distorted details, warped edges, blurry, low quality, unnatural shadows"


def _contains_any(text: str, values: List[str]) -> bool:
    return any(value in text for value in values)


def detect_action(text: str) -> str:
    lowered = text.lower()
    for action, hints in ACTION_HINTS.items():
        if _contains_any(lowered, hints):
            return action
    return "enhance"


def detect_targets(text: str) -> List[str]:
    lowered = text.lower()
    found = [word for word in TARGET_WORDS if word in lowered]
    return sorted(set(found)) or ["selected area"]


def detect_colors(text: str) -> List[str]:
    lowered_words = set(text.lower().replace(",", " ").replace(".", " ").split())
    return sorted(lowered_words.intersection(COLOR_WORDS))


def build_prompt(user_instruction: str, action: str, targets: List[str], image_context: Optional[str] = None) -> str:
    parts = []
    if image_context:
        parts.append(image_context.strip())
    parts.append(PRESET_PROMPTS.get(action, PRESET_PROMPTS["enhance"]))
    parts.append("target: " + ", ".join(targets))
    colors = detect_colors(user_instruction)
    if colors:
        parts.append("requested color: " + ", ".join(colors))
    parts.append("instruction: " + user_instruction.strip())
    return ", ".join(part for part in parts if part)


def build_edit_intent(user_instruction: str, image_context: Optional[str] = None) -> EditIntent:
    instruction = user_instruction.strip()
    if not instruction:
        raise ValueError("user_instruction is required")

    action = detect_action(instruction)
    targets = detect_targets(instruction)
    mode = "inpaint" if action in {"remove", "recolor", "replace"} else "enhance"
    style = "realistic"
    prompt = build_prompt(instruction, action, targets, image_context)
    notes = [
        "Use a mask around the selected target before running inpaint.",
        "Keep the generated prompt editable so the user can correct intent quickly.",
    ]

    return EditIntent(
        user_instruction=instruction,
        action=action,
        targets=targets,
        style=style,
        mode=mode,
        prompt=prompt,
        negative_prompt=NEGATIVE_BASE,
        notes=notes,
    )
