from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureRecipe:
    key: str
    label: str
    fooocus_area: str
    use_when: str
    avoid_when: str
    required_inputs: List[str]
    best_settings: List[str]
    prompt_strategy: str
    negative_strategy: str
    friction_notes: List[str]
    risk_notes: List[str]

    def as_markdown(self) -> str:
        return (
            f"### {self.label}\n"
            f"**Fooocus area:** {self.fooocus_area}\n\n"
            f"**Use when:** {self.use_when}\n\n"
            f"**Avoid when:** {self.avoid_when}\n\n"
            f"**Required inputs:** {', '.join(self.required_inputs) or 'None'}\n\n"
            f"**Best settings:**\n" + "\n".join([f"- {x}" for x in self.best_settings]) + "\n\n"
            f"**Prompt strategy:** {self.prompt_strategy}\n\n"
            f"**Negative strategy:** {self.negative_strategy}\n\n"
            f"**Friction notes:**\n" + "\n".join([f"- {x}" for x in self.friction_notes]) + "\n\n"
            f"**Risk notes:**\n" + "\n".join([f"- {x}" for x in self.risk_notes])
        )


FEATURE_CATALOG: Dict[str, FeatureRecipe] = {
    "text_to_image": FeatureRecipe(
        key="text_to_image",
        label="Text to Image",
        fooocus_area="Prompt only",
        use_when="The user wants a brand-new image and does not need an uploaded image preserved.",
        avoid_when="The user wants to keep identity, preserve a real face, edit only part of an image, or follow a specific pose/reference.",
        required_inputs=[],
        best_settings=["Speed for drafts", "Quality for finals", "2 images per run", "Use random seed for exploration, lock seed for refinement"],
        prompt_strategy="Describe one concrete image, not a list of images. Include subject, setting, wardrobe, camera/framing, lighting, and style.",
        negative_strategy="Block blur, artifacts, bad anatomy, bad hands, low quality, and unwanted style drift.",
        friction_notes=["Do not ask the user for masks or references when none are needed.", "Do not combine multiple shots into one prompt."],
        risk_notes=["Text prompts can drift from the user's intent if the scene is too broad."],
    ),
    "image_prompt": FeatureRecipe(
        key="image_prompt",
        label="Image Prompt Reference",
        fooocus_area="Input Image > Image Prompt",
        use_when="The uploaded image should inspire the new image through style, subject, composition, or loose identity similarity.",
        avoid_when="The user expects exact pixel-level edits or exact preservation of clothing/background.",
        required_inputs=["reference image"],
        best_settings=["Use ImagePrompt for loose reference", "Use FaceSwap/face reference for stronger identity", "Use 2-4 outputs", "Keep styles realistic for personal photos"],
        prompt_strategy="Say what should be preserved from the reference and what should change in the new image.",
        negative_strategy="Add different person, changed identity, distorted face, artifacts, bad hands when identity matters.",
        friction_notes=["Label this as a new image inspired by the reference, not an exact edit.", "Use shot-by-shot prompts for bundles."],
        risk_notes=["Identity and body can drift, especially with only one headshot."],
    ),
    "face_reference": FeatureRecipe(
        key="face_reference",
        label="Face Reference / FaceSwap Control",
        fooocus_area="Input Image > Image Prompt > FaceSwap type",
        use_when="The person should remain recognizable in a new image, new outfit, new scene, or photo bundle.",
        avoid_when="The user wants to edit the same original image area without changing the rest of the image.",
        required_inputs=["clear face reference"],
        best_settings=["Clear face photo first", "Add full-body pose reference separately for full-body outputs", "Use Fooocus Photograph + Enhance + Sharp", "Generate 2 candidates per shot"],
        prompt_strategy="Start with same adult person from the reference photo, preserve facial identity, age, skin tone, face shape, natural expression.",
        negative_strategy="different person, changed identity, distorted face, unnatural skin, distorted eyes, bad anatomy.",
        friction_notes=["Tell the user when a single headshot is not enough for body accuracy.", "Keep identity prompt consistent across shot prompts."],
        risk_notes=["Identity-sensitive workflow; consent and privacy should be considered."],
    ),
    "pyracanny": FeatureRecipe(
        key="pyracanny",
        label="PyraCanny Structure Control",
        fooocus_area="Input Image > Image Prompt > PyraCanny type",
        use_when="The output should follow hard edges, outlines, pose, body silhouette, or object layout from a reference image.",
        avoid_when="The user only wants broad vibe/style or wants softer composition control.",
        required_inputs=["structure or pose reference image"],
        best_settings=["Use when pose/edges matter", "Lower weight if the output follows the reference too rigidly", "Pair with face reference when identity matters"],
        prompt_strategy="Describe the final desired image while saying the uploaded image controls pose/structure.",
        negative_strategy="bad anatomy, warped body, broken pose, extra limbs, bad hands, distorted face.",
        friction_notes=["Explain that PyraCanny follows edges more strongly than CPDS.", "Use a clean reference image for best results."],
        risk_notes=["Too much structure weight can make outputs stiff or copied-looking."],
    ),
    "cpds": FeatureRecipe(
        key="cpds",
        label="CPDS Composition Control",
        fooocus_area="Input Image > Image Prompt > CPDS type",
        use_when="The output should follow broad composition, depth, pose, or camera framing without copying every edge.",
        avoid_when="The user needs exact edge/outline control.",
        required_inputs=["composition or pose reference image"],
        best_settings=["Use for softer pose/composition guidance", "Pair with face reference for identity", "Generate multiple candidates"],
        prompt_strategy="Describe the final desired image while saying the uploaded image controls composition/depth.",
        negative_strategy="bad anatomy, odd perspective, warped composition, distorted body, distorted face.",
        friction_notes=["Recommend CPDS before PyraCanny when the user wants natural-looking composition transfer.", "Use PyraCanny only when shape accuracy matters more."],
        risk_notes=["Composition may drift; generate candidates and compare."],
    ),
    "upscale_variation": FeatureRecipe(
        key="upscale_variation",
        label="Upscale / Variation",
        fooocus_area="Input Image > Upscale or Variation",
        use_when="The image is already mostly right and needs more resolution, polish, or small creative variation.",
        avoid_when="The user wants exact local editing or strong identity transformation.",
        required_inputs=["source image"],
        best_settings=["Upscale 2x for cleaner/larger output", "Vary Subtle for small changes", "Vary Strong for a noticeably different version"],
        prompt_strategy="Use preservation language: preserve identity, composition, lighting, and original look unless variation is intended.",
        negative_strategy="changed identity, over-smoothed skin, artifacts, blurry, warped details.",
        friction_notes=["Make the choice obvious: upscale = keep same, variation = create alternative.", "Do not use variation for exact edits."],
        risk_notes=["Variation can drift identity, clothing, and background."],
    ),
    "inpaint": FeatureRecipe(
        key="inpaint",
        label="Inpaint / Outpaint",
        fooocus_area="Input Image > Inpaint or Outpaint",
        use_when="The user wants to change only part of an uploaded image while preserving everything else.",
        avoid_when="The user wants a new scene, full-body generation from headshot, or a photo bundle.",
        required_inputs=["source image", "mask or auto-mask target"],
        best_settings=["Use Modify Content for replacement", "Use Improve Detail for polish", "Review mask before generation", "Keep prompt focused on masked area"],
        prompt_strategy="Say preserve unmasked areas, same identity, same lighting, same camera angle, only change the masked region.",
        negative_strategy="visible edit seams, changed identity, altered background, warped clothing, distorted face, artifacts.",
        friction_notes=["Auto-mask first, then let the user review the mask.", "Never hide mask review for exact edits."],
        risk_notes=["Highest-risk edit workflow because masks can target face/body/clothing."],
    ),
    "auto_mask_sam": FeatureRecipe(
        key="auto_mask_sam",
        label="Automatic Mask - SAM / GroundingDINO",
        fooocus_area="Input Image > Inpaint or Outpaint > Advanced Masking",
        use_when="The user wants one-click masking of a named object or region such as jacket, shirt, glasses, person, background.",
        avoid_when="The target is vague, very small, hidden, or requires broad person/background segmentation.",
        required_inputs=["source image", "detection prompt"],
        best_settings=["Use simple object words", "Review generated mask", "Use SAM for text-directed object masks"],
        prompt_strategy="Detection prompt should be short: jacket, shirt, glasses, background, person.",
        negative_strategy="Not applicable to detection, but generation negative should block seams and altered identity.",
        friction_notes=["The agent should generate detection prompt automatically.", "User should only approve or adjust the mask."],
        risk_notes=["Wrong mask means wrong edit; require visual review."],
    ),
    "auto_mask_u2net": FeatureRecipe(
        key="auto_mask_u2net",
        label="Automatic Mask - U2Net Segmentation",
        fooocus_area="Input Image > Inpaint or Outpaint > Advanced Masking",
        use_when="The user wants broad person/background segmentation instead of a text-directed object mask.",
        avoid_when="The user wants a precise small object like glasses or a specific clothing item.",
        required_inputs=["source image"],
        best_settings=["Use u2net_human_seg for person/background", "Use cloth segmentation when clothing regions are broad", "Review mask"],
        prompt_strategy="Use broad replacement prompts; do not ask U2Net for tiny object precision.",
        negative_strategy="visible seams, halo edges, cutout artifacts, changed identity.",
        friction_notes=["Pick U2Net when the target is whole person/background.", "Pick SAM when the target is a named item."],
        risk_notes=["Segmentation edges can create halos; enhance/upscale winners only."],
    ),
    "enhance": FeatureRecipe(
        key="enhance",
        label="Enhance",
        fooocus_area="Input Image > Enhance",
        use_when="The user already has a winner and wants polish, detail, or quality improvement.",
        avoid_when="The base image direction is wrong; fix composition/identity first.",
        required_inputs=["winner image"],
        best_settings=["Use after selecting winner", "Keep prompt conservative", "Avoid heavy style changes"],
        prompt_strategy="Preserve original image, improve details, natural skin texture, sharper finish, no identity changes.",
        negative_strategy="overprocessed, waxy skin, changed identity, artifacts, oversharpened, blurry.",
        friction_notes=["Enhance should be a final step, not first step.", "Never churn on bad candidates; regenerate direction first."],
        risk_notes=["Enhance can overprocess faces if pushed too hard."],
    ),
    "describe": FeatureRecipe(
        key="describe",
        label="Describe / Image-to-Prompt",
        fooocus_area="Input Image > Describe",
        use_when="The user wants help turning an uploaded image into a prompt or style description.",
        avoid_when="The user already knows exact changes or needs generation immediately.",
        required_inputs=["image to describe"],
        best_settings=["Use as research/planning", "Review generated text before applying"],
        prompt_strategy="Use description as draft, then rewrite into a clean generation prompt.",
        negative_strategy="Add negatives based on workflow after description is reviewed.",
        friction_notes=["Do not auto-apply descriptions without review.", "Keep this as a helper, not the main workflow."],
        risk_notes=["Descriptions can include unwanted sensitive or irrelevant details."],
    ),
}


def get_feature(key: str) -> FeatureRecipe:
    return FEATURE_CATALOG[key]


def list_features_markdown() -> str:
    return "\n\n".join(recipe.as_markdown() for recipe in FEATURE_CATALOG.values())


def feature_keys_for_workflow(workflow: str) -> List[str]:
    mapping = {
        "make new image": ["text_to_image"],
        "edit part of image": ["inpaint", "auto_mask_sam", "auto_mask_u2net", "enhance"],
        "keep identity": ["face_reference", "image_prompt", "cpds", "pyracanny", "upscale_variation"],
        "follow pose or shape": ["pyracanny", "cpds", "image_prompt"],
        "make bigger or cleaner": ["upscale_variation", "enhance"],
        "photo bundle": ["face_reference", "image_prompt", "cpds", "pyracanny", "upscale_variation", "enhance"],
    }
    return mapping.get(workflow, ["text_to_image"])


def feature_summary_for_workflow(workflow: str) -> str:
    keys = feature_keys_for_workflow(workflow)
    lines = []
    for key in keys:
        recipe = FEATURE_CATALOG[key]
        lines.append(f"- {recipe.label}: {recipe.use_when}")
    return "\n".join(lines)
