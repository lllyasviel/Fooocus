from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from local_markup.fooocus_feature_playbook import build_feature_reasoning, build_scenario_summary


WORKFLOW_LIBRARY: Dict[str, Dict[str, str]] = {
    "make new image": {
        "tool": "Text to Image",
        "fooocus_tab": "Prompt only",
        "why": "Best when no uploaded image needs to be preserved exactly.",
        "mask": "No mask needed.",
        "reference": "Optional style reference only.",
    },
    "edit part of image": {
        "tool": "Inpaint / Outpaint",
        "fooocus_tab": "Inpaint or Outpaint",
        "why": "Best when the user wants to change only part of an uploaded image.",
        "mask": "Use manual mask or automatic SAM/U2Net mask, then review before generation.",
        "reference": "Use the uploaded image as the source image.",
    },
    "keep identity": {
        "tool": "Image Prompt / Face Reference",
        "fooocus_tab": "Image Prompt",
        "why": "Best when the person should stay recognizable across new images.",
        "mask": "No mask unless doing exact edits.",
        "reference": "Use 2-5 clear photos: face close-up, upper body, full body if available.",
    },
    "follow pose or shape": {
        "tool": "PyraCanny / CPDS",
        "fooocus_tab": "Image Prompt",
        "why": "Best when a reference image should guide structure, pose, outline, or composition.",
        "mask": "No mask required. Use PyraCanny for edges, CPDS for composition/depth.",
        "reference": "Use a pose/structure reference image.",
    },
    "make bigger or cleaner": {
        "tool": "Upscale / Variation",
        "fooocus_tab": "Upscale or Variation",
        "why": "Best when the image is mostly right and needs more resolution or polish.",
        "mask": "No mask required.",
        "reference": "Use the image to upscale or vary.",
    },
    "photo bundle": {
        "tool": "Bundle Builder",
        "fooocus_tab": "Image Prompt",
        "why": "Best when the user wants a set of headshots, full-body portraits, lifestyle photos, or resort/fitness/business/casual variants.",
        "mask": "Use masks only for exact outfit/background edits. Use references for new bundle images.",
        "reference": "Use multiple reference photos for identity and body accuracy.",
    },
}


SHOT_PRESETS: Dict[str, List[str]] = {
    "photo bundle": [
        "standing full body near a modern resort pool, tasteful swimwear, relaxed confident posture, natural daylight, professional lifestyle photo",
        "walking beside a beach or resort walkway, tasteful resort look, candid but polished, realistic proportions, warm daylight",
        "upper body resort portrait near poolside seating, clean background, natural expression, realistic photography",
        "professional social profile portrait outdoors, modern lifestyle setting, sharp details, natural skin texture",
    ],
    "keep identity": [
        "single realistic portrait preserving identity, natural expression, professional lighting",
        "standing full body realistic portrait preserving identity, natural posture, realistic proportions",
        "upper body lifestyle photo preserving identity, clean background, sharp details",
        "outdoor candid-style portrait preserving identity, natural daylight, realistic photography",
    ],
    "edit part of image": [
        "exact edit of the selected masked area only, preserve every unmasked area, seamless realistic result",
        "same source image with a cleaner replacement in the masked region, no visible edit seams",
        "detail-focused inpaint of the selected area, natural texture and lighting match",
        "final polish pass preserving identity, pose, background, and camera angle",
    ],
    "follow pose or shape": [
        "follow the uploaded reference pose and composition, realistic photo result",
        "same composition with new styling, clean edges, natural lighting",
        "structure-guided variation with realistic proportions and sharp details",
        "composition-guided portrait with professional finish",
    ],
    "make bigger or cleaner": [
        "clean upscale preserving original identity, composition, texture, and lighting",
        "subtle improvement with sharper detail and fewer artifacts",
        "presentation-ready polished version, natural texture, no identity drift",
        "final high quality enhanced image preserving original look",
    ],
    "make new image": [
        "high quality realistic photo, natural lighting, sharp details",
        "professional portrait, clean composition, realistic skin texture",
        "lifestyle photo, natural expression, modern setting",
        "social profile image, realistic photography, polished finish",
    ],
}


@dataclass
class AgentPlan:
    user_goal: str
    selected_workflow: str
    tool: str
    fooocus_tab: str
    primary_prompt: str
    negative_prompt: str
    reference_strategy: str
    mask_strategy: str
    generation_settings: str
    next_steps: str
    warnings: str
    shot_prompts: List[str]
    best_first_shot: str
    handoff_recipe: str
    scenario_summary: str
    feature_reasoning: str

    def as_markdown(self) -> str:
        shot_text = "\n".join([f"{index + 1}. {prompt}" for index, prompt in enumerate(self.shot_prompts)])
        return (
            "## Agent Plan\n\n"
            f"**Goal:** {self.user_goal}\n\n"
            f"**Selected workflow:** {self.selected_workflow}\n\n"
            f"**Tool:** {self.tool}\n\n"
            f"**Fooocus area:** {self.fooocus_tab}\n\n"
            f"### Scenario summary\n{self.scenario_summary}\n\n"
            f"### Why this workflow\n{WORKFLOW_LIBRARY[self.selected_workflow]['why']}\n\n"
            f"### Reference strategy\n{self.reference_strategy}\n\n"
            f"### Mask strategy\n{self.mask_strategy}\n\n"
            f"### Recommended settings\n{self.generation_settings}\n\n"
            f"### Best first shot\n{self.best_first_shot}\n\n"
            f"### Shot prompts\n{shot_text}\n\n"
            f"### Fooocus hand-off recipe\n{self.handoff_recipe}\n\n"
            f"### Feature reasoning\n{self.feature_reasoning}\n\n"
            f"### Next steps\n{self.next_steps}\n\n"
            f"### Warnings / guardrails\n{self.warnings}"
        )


def _clean_goal(goal: str) -> str:
    return (goal or "").strip() or "Create a high quality realistic image."


def infer_workflow(goal: str, image_count: int, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool) -> str:
    text = goal.lower()
    if wants_bundle or any(word in text for word in ["bundle", "set of", "all types", "multiple images", "photoshoot", "photo shoot"]):
        return "photo bundle"
    if wants_exact_edit or any(word in text for word in ["remove", "replace", "change jacket", "change background", "edit only", "inpaint"]):
        return "edit part of image"
    if any(word in text for word in ["stand up", "standing", "full body", "same person", "identity", "me "]) or wants_identity:
        return "keep identity"
    if any(word in text for word in ["pose", "structure", "outline", "composition", "canny", "cpds"]):
        return "follow pose or shape"
    if any(word in text for word in ["upscale", "bigger", "enhance", "cleaner", "sharper"]):
        return "make bigger or cleaner"
    if image_count > 0:
        return "keep identity" if wants_identity else "edit part of image"
    return "make new image"


def _identity_prefix() -> str:
    return (
        "same adult man from the reference photo, preserve facial identity, age, skin tone, face shape, "
        "natural expression, realistic body proportions"
    )


def build_prompt(goal: str, workflow: str) -> str:
    goal = _clean_goal(goal)
    base = "high quality realistic photography, natural lighting, sharp details, professional finish"

    if workflow == "photo bundle":
        return f"{_identity_prefix()}, {SHOT_PRESETS['photo bundle'][0]}, user goal: {goal}, {base}"
    if workflow == "keep identity":
        return f"{_identity_prefix()}, {goal}, {base}"
    if workflow == "edit part of image":
        return "same person, preserve unchanged areas, seamless realistic edit, natural lighting, only change the requested area, " + goal
    if workflow == "follow pose or shape":
        return "use the uploaded reference for pose/composition guidance, " + goal + ", " + base
    if workflow == "make bigger or cleaner":
        return "improve image quality, preserve original identity and composition, sharper details, clean finish, " + goal
    return goal + ", " + base


def build_shot_prompts(goal: str, workflow: str) -> List[str]:
    goal = _clean_goal(goal)
    presets = SHOT_PRESETS.get(workflow, SHOT_PRESETS["make new image"])
    if workflow in {"photo bundle", "keep identity"}:
        return [f"{_identity_prefix()}, {shot}, user goal: {goal}" for shot in presets]
    if workflow == "edit part of image":
        return [f"same person, preserve unmasked areas, {shot}, user goal: {goal}" for shot in presets]
    return [f"{shot}, user goal: {goal}" for shot in presets]


def build_negative_prompt(workflow: str) -> str:
    common = "low quality, blurry, distorted face, bad hands, extra limbs, warped fingers, artifacts, unrealistic anatomy"
    if workflow in {"keep identity", "photo bundle", "edit part of image"}:
        return common + ", different person, changed identity, unnatural skin, distorted eyes, cropped head, missing legs"
    return common


def build_next_steps(workflow: str) -> str:
    if workflow == "edit part of image":
        return (
            "1. Upload the source image.\n"
            "2. Choose or generate a mask for the exact area.\n"
            "3. Review the mask before generating.\n"
            "4. Generate 2-4 candidates.\n"
            "5. Pick the best one and enhance/upscale if needed."
        )
    if workflow == "keep identity":
        return (
            "1. Upload 2-5 reference photos if available.\n"
            "2. Use a clear face reference first.\n"
            "3. Add full-body reference for standing or outfit accuracy.\n"
            "4. Generate multiple candidates and keep the closest identity match."
        )
    if workflow == "follow pose or shape":
        return (
            "1. Upload the structure or pose reference.\n"
            "2. Use PyraCanny for hard edges or CPDS for softer composition.\n"
            "3. Generate candidates and adjust weight if the output follows too much or too little."
        )
    if workflow == "make bigger or cleaner":
        return "1. Upload the image.\n2. Use Upscale (2x) for quality.\n3. Use subtle variation only when you want small creative changes."
    if workflow == "photo bundle":
        return (
            "1. Generate Shot 1 first.\n"
            "2. If identity is close, generate the next shot prompts one at a time.\n"
            "3. Keep the best identity match from each shot.\n"
            "4. Use upscale/enhance on winners."
        )
    return "1. Enter prompt.\n2. Pick style.\n3. Generate.\n4. Save the best result."


def build_handoff_recipe(workflow: str, image_count: int) -> str:
    if workflow == "photo bundle":
        return (
            "Fooocus tab: Image Prompt. Use Reference image 1 as FaceSwap/identity reference. "
            "If you have a full-body/pose reference, add it as a second Image Prompt reference using PyraCanny or CPDS. "
            "Generate one shot prompt at a time, 2 images per run, Speed for drafts, Quality for final."
        )
    if workflow == "keep identity":
        return (
            "Fooocus tab: Image Prompt. Use FaceSwap/face reference for the face photo. "
            "For standing or full-body images, add a second full-body pose reference if possible."
        )
    if workflow == "edit part of image":
        return "Fooocus tab: Inpaint or Outpaint. Upload source image, draw/review mask, paste prompt, generate."
    if workflow == "follow pose or shape":
        return "Fooocus tab: Image Prompt. Use PyraCanny for strong edge following or CPDS for softer composition following."
    if workflow == "make bigger or cleaner":
        return "Fooocus tab: Upscale or Variation. Use Upscale 2x for cleaner/larger output."
    return "Fooocus main prompt. Paste prompt and generate."


def build_agent_plan(
    goal: str,
    image_count: int = 0,
    wants_identity: bool = True,
    wants_exact_edit: bool = False,
    wants_bundle: bool = False,
) -> AgentPlan:
    goal = _clean_goal(goal)
    workflow = infer_workflow(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    info = WORKFLOW_LIBRARY[workflow]
    shot_prompts = build_shot_prompts(goal, workflow)
    primary_prompt = shot_prompts[0] if shot_prompts else build_prompt(goal, workflow)
    negative_prompt = build_negative_prompt(workflow)
    scenario_summary = build_scenario_summary(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    feature_reasoning = build_feature_reasoning(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    generation_settings = (
        "Performance: Speed for drafts, Quality for final.\n"
        "Image count: 2 candidates per shot.\n"
        "Styles: Fooocus Photograph + Fooocus Enhance + Fooocus Sharp for realistic personal photos.\n"
        "Seed: random for exploration, locked seed for refinement."
    )
    warnings = (
        "Reference images can drift if only one headshot is provided. Full-body goals work best with full-body references. "
        "Exact edits require a reviewed mask. For personal images, consider privacy/metadata settings."
    )
    return AgentPlan(
        user_goal=goal,
        selected_workflow=workflow,
        tool=info["tool"],
        fooocus_tab=info["fooocus_tab"],
        primary_prompt=primary_prompt,
        negative_prompt=negative_prompt,
        reference_strategy=info["reference"],
        mask_strategy=info["mask"],
        generation_settings=generation_settings,
        next_steps=build_next_steps(workflow),
        warnings=warnings,
        shot_prompts=shot_prompts,
        best_first_shot=primary_prompt,
        handoff_recipe=build_handoff_recipe(workflow, image_count),
        scenario_summary=scenario_summary,
        feature_reasoning=feature_reasoning,
    )


def build_agent_outputs(goal: str, image_1, image_2, image_3, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool):
    image_count = sum(x is not None for x in [image_1, image_2, image_3])
    plan = build_agent_plan(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    shot_list = "\n\n".join([f"Shot {index + 1}: {prompt}" for index, prompt in enumerate(plan.shot_prompts)])
    return (
        plan.as_markdown(),
        plan.primary_prompt,
        plan.negative_prompt,
        plan.tool,
        plan.fooocus_tab,
        shot_list,
        plan.handoff_recipe,
    )
