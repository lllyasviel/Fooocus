from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .ui_adapter_v2 import build_markup_ui_state


WORKFLOW_COPY = {
    "Generate New Image": {
        "tab": "Prompt only",
        "prompt_prefix": "Create a high quality SDXL image of",
        "steps": "30",
        "cfg": "4",
        "denoise": "n/a",
        "help": "Use this when you want a brand-new image. No input image is required.",
    },
    "Edit This Image": {
        "tab": "Inpaint or Outpaint",
        "prompt_prefix": "Change only the masked area to",
        "steps": "30 / 15",
        "cfg": "4",
        "denoise": "0.75 to 1.0",
        "help": "Use this when you want to change part of an uploaded image. Draw a mask over only the part to change.",
    },
    "Use Image as Reference": {
        "tab": "Image Prompt",
        "prompt_prefix": "Create a new image that keeps the reference style and subject, showing",
        "steps": "30",
        "cfg": "4",
        "denoise": "n/a",
        "help": "Use this when you want a new image inspired by an uploaded reference image, not an exact edit.",
    },
    "Improve / Enhance": {
        "tab": "Enhance",
        "prompt_prefix": "Improve quality, sharpness, lighting, and professional finish while preserving",
        "steps": "30 / 15",
        "cfg": "4",
        "denoise": "0.35 to 0.6",
        "help": "Use this when the image is basically right and you only want it cleaner or sharper.",
    },
}


AREA_HINTS = {
    "Face": "face",
    "Glasses": "glasses",
    "Hair": "hair",
    "Shirt / Clothes": "shirt, clothes",
    "Background": "background",
    "Object": "object",
    "Whole Image": "selected area",
    "I will draw the mask": "selected area",
}


@dataclass
class EasySdxlPlan:
    workflow: str
    recommended_tab: str
    positive_prompt: str
    inpaint_prompt: str
    detection_prompt: str
    negative_prompt: str
    steps: str
    cfg: str
    denoise: str
    mask_tip: str
    checklist: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _instruction_tokens(text: str) -> str:
    return (text or "").strip().lower()


def _infer_targets(area: str, instruction: str, fallback: str) -> List[str]:
    text = _instruction_tokens(instruction)
    targets: List[str] = []

    if area in AREA_HINTS and AREA_HINTS[area] != "selected area":
        targets.extend([part.strip() for part in AREA_HINTS[area].split(",")])

    keyword_targets = [
        ("glasses", ["glasses", "eyeglasses", "spectacles"]),
        ("shirt", ["shirt", "polo", "suit", "jacket", "tie", "clothes", "clothing", "outfit"]),
        ("face", ["face", "smile", "eyes", "skin", "head"]),
        ("hair", ["hair", "bald", "beard"]),
        ("background", ["background", "backdrop", "wall", "office", "street"]),
    ]

    for target, keywords in keyword_targets:
        if any(keyword in text for keyword in keywords):
            targets.append(target)

    if not targets and fallback:
        targets.extend([part.strip() for part in fallback.split(",") if part.strip()])

    deduped: List[str] = []
    for target in targets:
        if target and target not in deduped:
            deduped.append(target)
    return deduped or ["selected area"]


def _detection_prompt(targets: List[str]) -> str:
    cleaned = [target for target in targets if target and target != "selected area"]
    return ", ".join(cleaned) if cleaned else "selected area"


def _portrait_edit_prompt(instruction: str, targets: List[str]) -> str:
    text = _instruction_tokens(instruction)
    prompt_bits = [
        "same person",
        "preserve facial identity, age, skin tone, expression, camera angle, and lighting",
    ]

    if "glasses" in targets and any(word in text for word in ["remove", "without", "no "]):
        prompt_bits.append("without glasses")

    if "shirt" in targets:
        if "polo" in text:
            if "navy" in text or "blue" in text:
                prompt_bits.append("wearing a clean navy blue polo shirt")
            else:
                prompt_bits.append("wearing a clean polo shirt")
        elif "suit" in text and any(word in text for word in ["remove", "change", "replace"]):
            prompt_bits.append("replace suit jacket and tie with clean casual professional shirt")
        else:
            prompt_bits.append(instruction)
    elif instruction:
        prompt_bits.append(instruction)

    prompt_bits.extend([
        "professional realistic headshot",
        "natural fabric texture",
        "seamless edit",
        "no visible retouching artifacts",
    ])
    return ", ".join(prompt_bits)


def _negative_prompt(workflow: str, negative_prompt: str) -> str:
    base = negative_prompt or "artifacts, distorted details, warped edges, blurry, low quality, unnatural shadows"
    if workflow == "Edit This Image":
        additions = "changed identity, different person, deformed face, melted glasses, distorted eyes, extra collar, bad clothing seams"
        return f"{base}, {additions}"
    return base


def build_easy_sdxl_plan(
    workflow: str,
    area: str,
    user_instruction: str,
    image_context: Optional[str] = None,
) -> EasySdxlPlan:
    workflow = workflow if workflow in WORKFLOW_COPY else "Edit This Image"
    area = area if area in AREA_HINTS else "I will draw the mask"
    instruction = (user_instruction or "").strip()
    workflow_info = WORKFLOW_COPY[workflow]

    if not instruction:
        instruction = "describe the change you want"

    state = build_markup_ui_state(instruction, image_context=image_context)
    targets = _infer_targets(area, instruction, state.detection_prompt)
    detection = _detection_prompt(targets)

    if workflow == "Generate New Image":
        positive = f"{workflow_info['prompt_prefix']} {instruction}, high quality, detailed, sharp, professional"
        inpaint_prompt = ""
    elif workflow == "Use Image as Reference":
        positive = f"{workflow_info['prompt_prefix']} {instruction}, high quality, detailed, sharp, professional"
        inpaint_prompt = ""
    elif workflow == "Improve / Enhance":
        positive = f"{workflow_info['prompt_prefix']} {instruction}, natural details, clean lighting, sharp focus"
        inpaint_prompt = _portrait_edit_prompt(instruction, targets)
    else:
        positive = _portrait_edit_prompt(instruction, targets)
        inpaint_prompt = positive

    if workflow == "Edit This Image":
        mask_tip = (
            "Paint the white mask only over the changed areas. For your headshot example, mask the glasses and the suit/tie/clothes. "
            "Do not mask the whole face unless you want the face changed."
        )
    elif workflow == "Use Image as Reference":
        mask_tip = "Do not use this for exact edits. Use this only when the reference image should inspire a new image."
    elif workflow == "Improve / Enhance":
        mask_tip = "Use low denoise when preserving identity matters. Only mask details that need cleanup."
    else:
        mask_tip = "No mask needed. This creates a new image from the prompt."

    checklist = (
        f"Recommended workflow: {workflow}\n"
        f"Best Fooocus tab: {workflow_info['tab']}\n"
        f"What it means: {workflow_info['help']}\n\n"
        "Do this next:\n"
        f"1. Go to: {workflow_info['tab']}\n"
        f"2. Area to change/detect: {detection}\n"
        f"3. Mask tip: {mask_tip}\n"
        "4. Use the generated prompt below.\n"
        "5. Keep Advanced available for CFG, steps, denoise, seed, styles, and LoRAs."
    )

    return EasySdxlPlan(
        workflow=workflow,
        recommended_tab=workflow_info["tab"],
        positive_prompt=positive,
        inpaint_prompt=inpaint_prompt,
        detection_prompt=detection,
        negative_prompt=_negative_prompt(workflow, state.negative_prompt),
        steps=workflow_info["steps"],
        cfg=workflow_info["cfg"],
        denoise=workflow_info["denoise"],
        mask_tip=mask_tip,
        checklist=checklist,
    )


def build_easy_sdxl_outputs(workflow: str, area: str, user_instruction: str):
    plan = build_easy_sdxl_plan(workflow, area, user_instruction)
    settings = (
        f"Steps: {plan.steps}\n"
        f"CFG: {plan.cfg}\n"
        f"Denoise: {plan.denoise}\n"
        f"Recommended tab: {plan.recommended_tab}"
    )
    return (
        plan.positive_prompt,
        plan.inpaint_prompt,
        plan.detection_prompt,
        plan.negative_prompt,
        settings,
        plan.checklist,
    )
