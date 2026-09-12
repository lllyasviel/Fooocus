from __future__ import annotations

from dataclasses import dataclass
from typing import List

from local_markup.studio_knowledge import StudioFeature, route_feature


SHOT_PRESETS = {
    "text_to_image": [
        "high quality realistic photo, natural lighting, sharp details",
        "professional portrait, clean composition, realistic texture",
        "lifestyle photo, natural expression, modern setting",
        "social profile image, realistic photography, polished finish",
    ],
    "image_prompt": [
        "new realistic image inspired by the uploaded reference, natural lighting, sharp details",
        "reference-guided lifestyle photo, clean composition, professional finish",
        "reference-inspired portrait with realistic proportions and natural texture",
        "polished variation using the uploaded image as visual guidance",
    ],
    "face_reference": [
        "standing full-length resort lifestyle photo, relaxed confident posture, natural daylight, professional finish",
        "walking in a resort setting, polished casual look, realistic proportions, warm daylight",
        "upper-body resort portrait near poolside seating, clean background, natural expression, realistic photography",
        "professional social profile portrait outdoors, modern lifestyle setting, sharp details, natural texture",
    ],
    "pyracanny": [
        "follow the uploaded reference pose and edges, realistic photo result, clean structure",
        "same outline and pose with new styling, natural lighting, sharp details",
        "edge-guided variation with realistic proportions and professional finish",
        "structure-controlled portrait using the reference contours",
    ],
    "cpds": [
        "follow the uploaded reference composition and depth, realistic photo result",
        "same broad layout with new styling, natural lighting, polished finish",
        "composition-guided lifestyle image with realistic proportions",
        "depth-guided portrait with professional finish",
    ],
    "upscale": [
        "clean upscale preserving original composition, texture, and lighting",
        "subtle improvement with sharper detail and fewer artifacts",
        "presentation-ready polished version, natural texture, no drift",
        "final high quality enhanced image preserving original look",
    ],
    "variation": [
        "subtle new version inspired by the uploaded image, preserve the overall look",
        "creative variation inspired by the reference, realistic lighting and proportions",
        "alternate version with similar composition and polished finish",
        "new version using the source image as inspiration, natural texture",
    ],
    "inpaint": [
        "exact edit of the selected masked area only, preserve every unmasked area, seamless realistic result",
        "same source image with a cleaner replacement in the masked region, no visible edit seams",
        "detail-focused inpaint of the selected area, natural texture and lighting match",
        "final polish pass preserving pose, background, and camera angle",
    ],
    "auto_mask_sam": [
        "use automatic object mask, edit only the detected target, preserve all other areas",
        "SAM mask exact edit with seamless realistic replacement",
        "object-targeted inpaint with natural lighting and texture match",
        "reviewed automatic mask edit preserving unmasked regions",
    ],
    "auto_mask_u2net": [
        "broad subject or background segmentation edit with realistic finish",
        "segmentation-based cleanup with natural lighting and texture match",
        "background or subject mask edit with professional realistic finish",
        "reviewed broad segmentation mask result",
    ],
    "enhance": [
        "enhance details while preserving composition and natural texture",
        "polish winning image with sharper details and fewer artifacts",
        "professional finish pass, natural lighting, realistic texture",
        "final enhancement preserving original look",
    ],
    "describe": [
        "analyze uploaded image and create a reusable prompt after user review",
        "turn image into a clear prompt with style, subject, setting, and lighting",
        "describe visual features for prompt planning",
        "draft prompt from uploaded image for user review",
    ],
}


@dataclass
class AgentPlan:
    user_goal: str
    feature: StudioFeature
    primary_prompt: str
    negative_prompt: str
    generation_settings: str
    next_steps: str
    shot_prompts: List[str]
    handoff_recipe: str

    @property
    def tool(self) -> str:
        return self.feature.label

    @property
    def fooocus_tab(self) -> str:
        return self.feature.fooocus_area

    def as_markdown(self) -> str:
        shot_text = "\n".join([f"{index + 1}. {prompt}" for index, prompt in enumerate(self.shot_prompts)])
        controls = "\n".join([f"- {key}: {value}" for key, value in self.feature.controls.items()])
        return (
            "## Agent Plan\n\n"
            f"**Goal:** {self.user_goal}\n\n"
            f"**Selected feature:** {self.feature.label}\n\n"
            f"**Fooocus area:** {self.feature.fooocus_area}\n\n"
            f"### Use when\n{self.feature.use_when}\n\n"
            f"### Avoid when\n{self.feature.avoid_when}\n\n"
            f"### Required inputs\n{', '.join(self.feature.required_inputs)}\n\n"
            f"### Controls from single source of truth\n{controls}\n\n"
            f"### Best first shot\n{self.primary_prompt}\n\n"
            f"### Shot prompts\n{shot_text}\n\n"
            f"### Fooocus hand-off recipe\n{self.handoff_recipe}\n\n"
            f"### Recommended settings\n{self.generation_settings}\n\n"
            f"### Next steps\n{self.next_steps}"
        )


def clean_goal(goal: str) -> str:
    return (goal or "").strip() or "Create a high quality realistic image."


def subject_prefix() -> str:
    return "same authorized adult subject from the reference photos, consistent appearance, natural expression, realistic proportions"


def build_shot_prompts(goal: str, feature_key: str, wants_subject_consistency: bool) -> List[str]:
    goal = clean_goal(goal)
    presets = SHOT_PRESETS.get(feature_key, SHOT_PRESETS["text_to_image"])
    if wants_subject_consistency or feature_key == "face_reference":
        return [f"{subject_prefix()}, {shot}, user goal: {goal}" for shot in presets]
    if feature_key in {"inpaint", "auto_mask_sam", "auto_mask_u2net"}:
        return [f"preserve unmasked areas, {shot}, user goal: {goal}" for shot in presets]
    return [f"{shot}, user goal: {goal}" for shot in presets]


def build_negative_prompt(feature_key: str) -> str:
    common = "low quality, blurry, distorted details, bad hands, extra limbs, warped fingers, artifacts, unrealistic anatomy"
    if feature_key in {"face_reference", "image_prompt", "inpaint", "auto_mask_sam", "auto_mask_u2net", "variation"}:
        return common + ", inconsistent subject, changed appearance, unnatural skin, distorted eyes, cropped head, missing legs"
    return common


def build_handoff_recipe(feature: StudioFeature) -> str:
    key = feature.key
    if key == "face_reference":
        return "Fooocus area: Image Prompt. Use a clear authorized subject reference first. Add an optional pose/style reference for full-length or bundle outputs. Generate one shot prompt at a time."
    if key == "image_prompt":
        return "Fooocus area: Image Prompt. Upload reference image, use ImagePrompt control, paste the prompt and negative prompt."
    if key == "pyracanny":
        return "Fooocus area: Image Prompt. Upload structure reference, choose PyraCanny, then paste the structure-guided prompt."
    if key == "cpds":
        return "Fooocus area: Image Prompt. Upload composition reference, choose CPDS, then paste the composition-guided prompt."
    if key == "upscale":
        return "Fooocus area: Upscale or Variation. Upload image and choose Upscale 2x for the first pass."
    if key == "variation":
        return "Fooocus area: Upscale or Variation. Use Vary Subtle for small changes or Vary Strong for creative changes."
    if key == "inpaint":
        return "Fooocus area: Inpaint or Outpaint. Upload source image, draw/review mask, paste inpaint prompt, generate."
    if key == "auto_mask_sam":
        return "Fooocus area: Inpaint or Outpaint. Enable Advanced Masking, choose SAM, enter a narrow detection prompt, generate mask, review it, then generate."
    if key == "auto_mask_u2net":
        return "Fooocus area: Inpaint or Outpaint. Enable Advanced Masking, choose a U2Net segmentation model, review mask, then generate."
    if key == "enhance":
        return "Fooocus area: Enhance. Upload the winning image and enhance only after the direction is already correct."
    if key == "describe":
        return "Fooocus area: Describe. Generate a draft description, review it, then use it as a prompt only if it matches the intent."
    return "Fooocus main prompt. Paste prompt and generate."


def build_next_steps(feature: StudioFeature) -> str:
    key = feature.key
    if key in {"inpaint", "auto_mask_sam", "auto_mask_u2net"}:
        return "1. Upload source image.\n2. Create or review mask.\n3. Confirm only the target area is selected.\n4. Generate 2 candidates.\n5. Upscale/enhance only the winner."
    if key in {"face_reference", "image_prompt", "pyracanny", "cpds"}:
        return "1. Upload reference images.\n2. Copy the best first shot prompt.\n3. Generate 2 candidates.\n4. Keep the closest match.\n5. Continue with the next shot prompt."
    if key in {"upscale", "variation", "enhance"}:
        return "1. Upload the image.\n2. Run one conservative pass.\n3. Compare before/after.\n4. Save the best version."
    if key == "describe":
        return "1. Upload the image.\n2. Generate description.\n3. Review and edit the prompt.\n4. Use the reviewed prompt in the right workflow."
    return "1. Paste prompt.\n2. Pick style.\n3. Generate.\n4. Save the best result."


def build_agent_plan(goal: str, image_count: int = 0, wants_identity: bool = True, wants_exact_edit: bool = False, wants_bundle: bool = False) -> AgentPlan:
    goal = clean_goal(goal)
    feature = route_feature(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    shot_prompts = build_shot_prompts(goal, feature.key, wants_identity or feature.key == "face_reference")
    return AgentPlan(
        user_goal=goal,
        feature=feature,
        primary_prompt=shot_prompts[0],
        negative_prompt=build_negative_prompt(feature.key),
        generation_settings=(
            "Performance: Speed for drafts, Quality for final.\n"
            "Image count: 2 candidates per shot.\n"
            "Styles: Fooocus Photograph + Fooocus Enhance + Fooocus Sharp for realistic personal photos.\n"
            "Seed: random for exploration, locked seed for refinement."
        ),
        next_steps=build_next_steps(feature),
        shot_prompts=shot_prompts,
        handoff_recipe=build_handoff_recipe(feature),
    )


def build_agent_outputs(goal: str, image_1, image_2, image_3, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool):
    image_count = sum(x is not None for x in [image_1, image_2, image_3])
    plan = build_agent_plan(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    shot_list = "\n\n".join([f"Shot {index + 1}: {prompt}" for index, prompt in enumerate(plan.shot_prompts)])
    return plan.as_markdown(), plan.primary_prompt, plan.negative_prompt, plan.tool, plan.fooocus_tab, shot_list, plan.handoff_recipe
