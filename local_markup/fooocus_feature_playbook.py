from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureCard:
    name: str
    category: str
    use_when: str
    avoid_when: str
    fooocus_area: str
    key_settings: str
    user_action: str
    risk_note: str


FEATURE_PLAYBOOK: Dict[str, FeatureCard] = {
    "text_to_image": FeatureCard(
        name="Text to Image",
        category="Create",
        use_when="The user wants a new image from words and does not need to preserve an uploaded image.",
        avoid_when="The user needs the same person, same pose, exact clothing edit, or exact background edit.",
        fooocus_area="Main prompt / Prompt only",
        key_settings="Use Quality for final, Speed for drafts. Use realistic styles for people/photo work.",
        user_action="Enter prompt, choose styles, generate candidates.",
        risk_note="Identity is invented unless reference images are used.",
    ),
    "image_prompt": FeatureCard(
        name="Image Prompt",
        category="Reference",
        use_when="The user wants a new image inspired by uploaded references, style, composition, or identity.",
        avoid_when="The user wants to change only one exact part of the original image; use Inpaint instead.",
        fooocus_area="Input Image > Image Prompt",
        key_settings="Use ImagePrompt for general visual reference. Adjust stop/weight if output follows too much or too little.",
        user_action="Upload reference image, choose ImagePrompt type, paste prompt, generate.",
        risk_note="The output can drift from the original image; call it a new image, not an exact edit.",
    ),
    "faceswap": FeatureCard(
        name="FaceSwap / Face Reference",
        category="Identity",
        use_when="The user wants the same adult person to remain recognizable in a new image.",
        avoid_when="The user only wants to fix a small face detail in the same photo; use Inpaint or Enhance instead.",
        fooocus_area="Input Image > Image Prompt",
        key_settings="Use a clear face image as the first reference. Add body/pose reference separately for full-body outputs.",
        user_action="Upload clear face reference, set reference type to FaceSwap/face reference, use identity-preserving prompt.",
        risk_note="Identity-sensitive. Requires user consent/ownership of reference images.",
    ),
    "pyracanny": FeatureCard(
        name="PyraCanny",
        category="Structure",
        use_when="The user wants the generated image to follow strong edges, outlines, pose, clothing silhouette, or layout.",
        avoid_when="The user wants loose composition only; CPDS is usually better.",
        fooocus_area="Input Image > Image Prompt",
        key_settings="Use PyraCanny when edges and pose matter. Lower weight if it over-constrains the image.",
        user_action="Upload pose/structure reference and choose PyraCanny.",
        risk_note="Can copy unwanted shapes from the reference if weight is too strong.",
    ),
    "cpds": FeatureCard(
        name="CPDS",
        category="Composition",
        use_when="The user wants broad composition, depth, pose, or camera layout without hard edge copying.",
        avoid_when="The user needs exact lines, outlines, or object boundaries; use PyraCanny.",
        fooocus_area="Input Image > Image Prompt",
        key_settings="Use CPDS for softer composition/depth guidance. Good second reference for full-body/pose goals.",
        user_action="Upload composition or pose reference and choose CPDS.",
        risk_note="May not preserve exact details; it guides layout more than identity.",
    ),
    "upscale": FeatureCard(
        name="Upscale",
        category="Polish",
        use_when="The image is already good and needs to be larger, cleaner, or more finished.",
        avoid_when="The image needs a major content change; use Inpaint, Image Prompt, or Variation first.",
        fooocus_area="Input Image > Upscale or Variation",
        key_settings="Use Upscale 2x for final winners. Use after selecting the best candidate.",
        user_action="Upload final candidate, select Upscale, generate.",
        risk_note="Upscaling can amplify artifacts if the source is weak.",
    ),
    "variation_subtle": FeatureCard(
        name="Subtle Variation",
        category="Explore",
        use_when="The user likes an image and wants a small realistic alternative.",
        avoid_when="The user wants an exact edit; use Inpaint.",
        fooocus_area="Input Image > Upscale or Variation",
        key_settings="Use Vary Subtle to preserve most of the image.",
        user_action="Upload image, choose Vary Subtle, generate.",
        risk_note="Identity/details can drift; not a precision editor.",
    ),
    "variation_strong": FeatureCard(
        name="Strong Variation",
        category="Explore",
        use_when="The user wants a noticeably different version inspired by the uploaded image.",
        avoid_when="The user wants identity or clothing preserved exactly.",
        fooocus_area="Input Image > Upscale or Variation",
        key_settings="Use Vary Strong when creative drift is acceptable.",
        user_action="Upload image, choose Vary Strong, generate candidates.",
        risk_note="Expect drift in face, body, clothing, and background.",
    ),
    "inpaint": FeatureCard(
        name="Inpaint",
        category="Exact Edit",
        use_when="The user wants to change one part of an existing image while preserving the rest.",
        avoid_when="The user wants a brand new image or full scene rewrite; Image Prompt or text-to-image may be better.",
        fooocus_area="Input Image > Inpaint or Outpaint",
        key_settings="Use Modify Content for changing clothing/background/object. Review mask carefully.",
        user_action="Upload source image, draw or generate mask, add inpaint prompt, generate.",
        risk_note="Mask quality determines result quality. Bad masks cause broken edits.",
    ),
    "outpaint": FeatureCard(
        name="Outpaint",
        category="Expand",
        use_when="The user wants to extend the image beyond its current edges.",
        avoid_when="The user only wants to change the subject; use Inpaint.",
        fooocus_area="Input Image > Inpaint or Outpaint",
        key_settings="Choose left/right/top/bottom expansion and keep prompt consistent with scene.",
        user_action="Upload image, choose outpaint directions, generate.",
        risk_note="Expanded areas may invent new context; review for consistency.",
    ),
    "sam_mask": FeatureCard(
        name="SAM / GroundingDINO Mask",
        category="Masking",
        use_when="The user wants automatic text-directed masks such as jacket, shirt, face, glasses, person, background, car, object.",
        avoid_when="The target is too vague or needs pixel-perfect manual control.",
        fooocus_area="Input Image > Inpaint or Outpaint > Advanced Masking",
        key_settings="Use detection prompt with concrete nouns. Review generated mask before generation.",
        user_action="Enable Advanced Masking, choose SAM, enter detection prompt, generate mask, review.",
        risk_note="Detection prompt must match visible objects. SAM can mask too much or too little.",
    ),
    "u2net_mask": FeatureCard(
        name="U2Net Mask",
        category="Masking",
        use_when="The user wants broad person/background/clothing segmentation without a text-directed object prompt.",
        avoid_when="The user needs a specific object like glasses or a small logo; SAM is better.",
        fooocus_area="Input Image > Inpaint or Outpaint > Advanced Masking",
        key_settings="Use u2net_human_seg for person/background; use clothing segmentation when available for clothes.",
        user_action="Enable Advanced Masking, choose U2Net-family model, generate mask, review.",
        risk_note="Broad masks may include areas that should stay unchanged.",
    ),
    "enhance": FeatureCard(
        name="Enhance",
        category="Polish",
        use_when="The image is close but needs detail improvement, face/detail cleanup, or final polish.",
        avoid_when="The content is wrong; fix content first with Inpaint/Image Prompt.",
        fooocus_area="Input Image > Enhance",
        key_settings="Use after choosing a winner. Avoid over-enhancing faces and skin.",
        user_action="Upload winner, choose enhance settings, generate final.",
        risk_note="Can over-sharpen or change identity if pushed too far.",
    ),
    "describe": FeatureCard(
        name="Describe / Interrogate",
        category="Understand",
        use_when="The user wants the app to explain an uploaded image or convert it into a useful prompt.",
        avoid_when="The generated description may be pasted without review into sensitive workflows.",
        fooocus_area="Input Image > Describe",
        key_settings="Use to draft a prompt, then edit before generating.",
        user_action="Upload image, run describe, review output, then use edited prompt.",
        risk_note="Descriptions can include unwanted details; review before applying.",
    ),
    "styles": FeatureCard(
        name="Styles",
        category="Look",
        use_when="The user wants a specific aesthetic: photograph, cinematic, anime, ad, real estate, luxury, etc.",
        avoid_when="Exact identity edits where style drift may change the person.",
        fooocus_area="Advanced > Styles",
        key_settings="For realistic people use Fooocus Photograph, Enhance, Sharp. Avoid Random Style for exact edits.",
        user_action="Pick styles that match the goal; keep exact-edit styles conservative.",
        risk_note="Styles modify prompts and can change identity, realism, or mood.",
    ),
    "loras": FeatureCard(
        name="LoRAs",
        category="Model Control",
        use_when="The user wants a trained style, character, product, or specialized look loaded through a LoRA.",
        avoid_when="The LoRA is unknown, too strong, or conflicts with identity preservation.",
        fooocus_area="Advanced > Models",
        key_settings="Use low-to-moderate weights first. Log LoRA names for repeatability.",
        user_action="Select LoRA and weight, generate tests, adjust weight.",
        risk_note="LoRAs can dominate the image and override identity/style.",
    ),
    "seed": FeatureCard(
        name="Seed Control",
        category="Repeatability",
        use_when="The user likes a result and wants controlled refinement.",
        avoid_when="The user is still exploring broad directions.",
        fooocus_area="Advanced > Settings",
        key_settings="Use random seed for exploration, lock seed for refinement.",
        user_action="After a good result, reuse seed and change one setting at a time.",
        risk_note="Changing too many settings with a locked seed makes debugging harder.",
    ),
    "privacy_metadata": FeatureCard(
        name="Metadata / Privacy",
        category="Safety",
        use_when="The user is working with personal images, client images, sensitive prompts, or private identity references.",
        avoid_when="Never ignore privacy for personal/identity workflows.",
        fooocus_area="Settings / Logs / Output metadata",
        key_settings="Consider disabling metadata/logging for private work when needed.",
        user_action="Review output folder/log behavior and metadata settings before sharing images.",
        risk_note="Prompts, seeds, and model settings can be stored in logs/metadata.",
    ),
}


SCENARIO_RULES = [
    ("remove_jacket", ["remove jacket", "take off jacket", "without jacket", "replace jacket", "jacket off"], ["inpaint", "sam_mask", "styles", "seed", "upscale"]),
    ("remove_glasses", ["remove glasses", "no glasses", "without glasses"], ["inpaint", "sam_mask", "enhance", "seed"]),
    ("change_background", ["change background", "replace background", "new background"], ["inpaint", "sam_mask", "u2net_mask", "styles"]),
    ("stand_full_body", ["stand", "standing", "full body", "full-body", "whole body"], ["faceswap", "cpds", "pyracanny", "image_prompt", "styles", "seed"]),
    ("same_person", ["same person", "keep my face", "recognizable", "identity", "looks like me"], ["faceswap", "image_prompt", "styles", "seed"]),
    ("pose_reference", ["pose", "body pose", "stance", "composition", "layout"], ["cpds", "pyracanny", "image_prompt"]),
    ("upscale_clean", ["upscale", "larger", "bigger", "cleaner", "sharper", "enhance quality"], ["upscale", "enhance", "seed"]),
    ("variation", ["variation", "different version", "another version", "creative version"], ["variation_subtle", "variation_strong", "image_prompt"]),
    ("prompt_from_image", ["describe", "what is in image", "turn image into prompt", "image to prompt"], ["describe", "styles"]),
    ("bundle", ["bundle", "photoshoot", "photo shoot", "set of", "all types", "multiple images"], ["faceswap", "image_prompt", "cpds", "styles", "seed", "upscale"]),
]


def match_feature_ids(goal: str, image_count: int, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool) -> List[str]:
    text = (goal or "").lower()
    selected: List[str] = []

    for _rule_name, terms, feature_ids in SCENARIO_RULES:
        if any(term in text for term in terms):
            selected.extend(feature_ids)

    if wants_bundle:
        selected.extend(["faceswap", "image_prompt", "cpds", "styles", "seed", "upscale"])
    if wants_exact_edit:
        selected.extend(["inpaint", "sam_mask", "u2net_mask", "seed"])
    if wants_identity:
        selected.extend(["faceswap", "image_prompt", "styles"])
    if image_count == 0 and not selected:
        selected.append("text_to_image")
    if image_count > 0 and not selected:
        selected.extend(["image_prompt", "describe"])

    selected.append("privacy_metadata")

    deduped: List[str] = []
    for feature_id in selected:
        if feature_id in FEATURE_PLAYBOOK and feature_id not in deduped:
            deduped.append(feature_id)
    return deduped


def build_feature_reasoning(goal: str, image_count: int, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool) -> str:
    feature_ids = match_feature_ids(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    lines: List[str] = []
    for feature_id in feature_ids:
        feature = FEATURE_PLAYBOOK[feature_id]
        lines.append(
            f"### {feature.name}\n"
            f"Category: {feature.category}\n"
            f"Use when: {feature.use_when}\n"
            f"Avoid when: {feature.avoid_when}\n"
            f"Fooocus area: {feature.fooocus_area}\n"
            f"Key settings: {feature.key_settings}\n"
            f"User action: {feature.user_action}\n"
            f"Risk note: {feature.risk_note}"
        )
    return "\n\n".join(lines)


def build_scenario_summary(goal: str, image_count: int, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool) -> str:
    feature_ids = match_feature_ids(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    names = [FEATURE_PLAYBOOK[feature_id].name for feature_id in feature_ids]
    return "Recommended feature stack: " + " → ".join(names)
