from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StudioFeature:
    key: str
    label: str
    fooocus_area: str
    use_when: str
    avoid_when: str
    required_inputs: List[str]
    outputs: List[str]
    controls: Dict[str, str]
    notes: str


FEATURES: Dict[str, StudioFeature] = {
    "text_to_image": StudioFeature(
        key="text_to_image",
        label="Text to Image",
        fooocus_area="Prompt only",
        use_when="The user wants a brand-new image and does not need a source image preserved.",
        avoid_when="The user wants exact edits to an uploaded image or subject consistency from a reference.",
        required_inputs=["plain prompt"],
        outputs=["primary prompt", "negative prompt", "style recommendation"],
        controls={"input_image": "off", "mask": "none", "styles": "optional"},
        notes="Best first path for clean new-image requests.",
    ),
    "image_prompt": StudioFeature(
        key="image_prompt",
        label="Image Prompt reference",
        fooocus_area="Image Prompt",
        use_when="The uploaded image should inspire a new image, style, composition, or visual direction.",
        avoid_when="The user expects a local masked edit to only one part of the source image.",
        required_inputs=["reference image", "main prompt"],
        outputs=["reference strategy", "prompt", "negative prompt"],
        controls={"input_image": "on", "control_type": "ImagePrompt", "mask": "none"},
        notes="Creates a new image guided by the reference; it is not an exact editor.",
    ),
    "face_reference": StudioFeature(
        key="face_reference",
        label="Face / Subject Reference",
        fooocus_area="Image Prompt",
        use_when="The subject should stay consistent across a new image or a set of planned shots.",
        avoid_when="The user only wants a minor local edit; use inpaint instead.",
        required_inputs=["clear subject reference", "optional upper-body reference", "optional full-body or pose reference"],
        outputs=["subject-consistency prompt", "shot plan", "handoff recipe"],
        controls={"input_image": "on", "control_type": "FaceSwap or face reference", "mask": "none"},
        notes="Use one focused shot prompt at a time; weak references may drift on full-body outputs.",
    ),
    "pyracanny": StudioFeature(
        key="pyracanny",
        label="PyraCanny structure control",
        fooocus_area="Image Prompt",
        use_when="The output should follow hard edges, pose, outlines, or strong visual structure from a reference image.",
        avoid_when="The user wants loose composition only; use CPDS instead.",
        required_inputs=["structure or pose reference image", "main prompt"],
        outputs=["structure-control prompt", "control guidance"],
        controls={"input_image": "on", "control_type": "PyraCanny", "mask": "none"},
        notes="High structure weight can over-constrain the result.",
    ),
    "cpds": StudioFeature(
        key="cpds",
        label="CPDS composition control",
        fooocus_area="Image Prompt",
        use_when="The output should follow broad composition, depth, or layout without copying every edge.",
        avoid_when="The user needs exact edge or line following; use PyraCanny instead.",
        required_inputs=["composition reference image", "main prompt"],
        outputs=["composition-control prompt", "control guidance"],
        controls={"input_image": "on", "control_type": "CPDS", "mask": "none"},
        notes="Good for broad layout/depth transfer.",
    ),
    "upscale": StudioFeature(
        key="upscale",
        label="Upscale image",
        fooocus_area="Upscale or Variation",
        use_when="The image direction is already good and needs more size, clarity, or polish.",
        avoid_when="The user wants a major creative change; use variation or image prompt.",
        required_inputs=["source image"],
        outputs=["upscale plan", "artifact-focused negative prompt"],
        controls={"input_image": "on", "uov_method": "Upscale 2x", "mask": "none"},
        notes="Use before enhancement when the image direction is already correct.",
    ),
    "variation": StudioFeature(
        key="variation",
        label="Variation",
        fooocus_area="Upscale or Variation",
        use_when="The user wants a similar but different version inspired by the uploaded image.",
        avoid_when="The user wants exact local edits or strict subject preservation.",
        required_inputs=["source image", "variation strength"],
        outputs=["variation prompt", "drift expectation"],
        controls={"input_image": "on", "uov_method": "Vary Subtle or Vary Strong", "mask": "none"},
        notes="Label this as inspired-by, not exact edit.",
    ),
    "inpaint": StudioFeature(
        key="inpaint",
        label="Inpaint / Outpaint exact edit",
        fooocus_area="Inpaint or Outpaint",
        use_when="The user wants to change one part of an uploaded image while preserving the rest.",
        avoid_when="The user wants a brand-new scene from references; use image prompt instead.",
        required_inputs=["source image", "mask", "inpaint prompt"],
        outputs=["mask strategy", "inpaint prompt", "negative prompt"],
        controls={"input_image": "on", "inpaint_mode": "Modify Content", "mask": "manual or automatic"},
        notes="Mask review is part of the handoff before generation.",
    ),
    "auto_mask_sam": StudioFeature(
        key="auto_mask_sam",
        label="SAM / GroundingDINO automatic mask",
        fooocus_area="Inpaint or Outpaint",
        use_when="The user wants text-directed automatic masking for a specific object, clothing item, person, or background.",
        avoid_when="The target is broad or vague; use manual masking or U2Net segmentation.",
        required_inputs=["source image", "detection prompt"],
        outputs=["detection prompt", "mask review checklist"],
        controls={"input_image": "on", "advanced_masking": "on", "mask_model": "SAM with detection prompt"},
        notes="Detection prompt should be narrow and mask should be reviewed.",
    ),
    "auto_mask_u2net": StudioFeature(
        key="auto_mask_u2net",
        label="U2Net segmentation mask",
        fooocus_area="Inpaint or Outpaint",
        use_when="The user wants broad subject, background, or clothing segmentation rather than a narrow object mask.",
        avoid_when="The user needs a very specific object selected; use SAM/GroundingDINO instead.",
        required_inputs=["source image"],
        outputs=["segmentation strategy", "mask review checklist"],
        controls={"input_image": "on", "advanced_masking": "on", "mask_model": "U2Net family"},
        notes="Broad masks can include too much; review before generation.",
    ),
    "enhance": StudioFeature(
        key="enhance",
        label="Enhance",
        fooocus_area="Enhance",
        use_when="The user wants to improve details, refine a winning image, or polish after generation.",
        avoid_when="The image direction is wrong; regenerate or vary before enhancing.",
        required_inputs=["source image", "enhance goal"],
        outputs=["enhance prompt", "settings recommendation"],
        controls={"input_image": "on", "enhance": "on", "mask": "optional"},
        notes="Use after a winning candidate exists.",
    ),
    "describe": StudioFeature(
        key="describe",
        label="Describe / image-to-prompt",
        fooocus_area="Describe",
        use_when="The user wants to understand an uploaded image or turn it into a reusable prompt.",
        avoid_when="The description will be reused without review.",
        required_inputs=["source image"],
        outputs=["draft prompt", "review checklist"],
        controls={"input_image": "on", "describe": "on", "apply_directly": "review first"},
        notes="Treat output as a draft prompt for review.",
    ),
    "styles": StudioFeature(
        key="styles",
        label="Styles",
        fooocus_area="Styles",
        use_when="The user needs visual language, photographic finish, or art direction guidance.",
        avoid_when="The style conflicts with the selected workflow or prompt intent.",
        required_inputs=["goal", "optional preferred style"],
        outputs=["style explanation", "style recommendation"],
        controls={"styles_tab": "select one or more styles", "prompt": "keep style aligned with goal"},
        notes="Use as support for most workflows, not as a standalone generation path.",
    ),
    "models_loras": StudioFeature(
        key="models_loras",
        label="Models / LoRAs",
        fooocus_area="Advanced model controls",
        use_when="The user needs specialized model or style behavior beyond normal styles.",
        avoid_when="The core workflow is not yet chosen or the model effect is unknown.",
        required_inputs=["model or LoRA name", "desired effect"],
        outputs=["model guidance", "LoRA guidance"],
        controls={"advanced": "model and LoRA selectors", "weights": "test conservatively"},
        notes="Later adapter work can turn this into model selection guidance.",
    ),
}


SCENARIO_RULES: List[tuple[str, List[str], str]] = [
    ("photo bundle", ["bundle", "set of", "all types", "photoshoot", "photo shoot", "multiple images"], "face_reference"),
    ("exact local edit", ["remove", "replace", "change only", "edit only", "inpaint", "mask"], "inpaint"),
    ("automatic object mask", ["auto mask", "detect", "scan", "select object", "select the object"], "auto_mask_sam"),
    ("background/person segmentation", ["person mask", "background mask", "segment", "segmentation", "select subject", "select background"], "auto_mask_u2net"),
    ("subject consistency", ["same person", "keep my face", "recognizable", "identity", "me standing", "make me", "standing lifestyle", "full body resort lifestyle"], "face_reference"),
    ("pose or edge control", ["pose", "outline", "edges", "canny", "structure"], "pyracanny"),
    ("composition control", ["composition", "depth", "layout", "cpds"], "cpds"),
    ("upscale", ["upscale", "bigger", "larger", "higher resolution", "sharper"], "upscale"),
    ("variation", ["variation", "new version", "inspired by", "similar but different"], "variation"),
    ("enhance", ["enhance", "improve", "polish", "clean up"], "enhance"),
    ("describe", ["describe", "what is in", "turn into prompt", "analyze image"], "describe"),
    ("style guidance", ["visual style", "style guidance", "visual language", "photo finish"], "styles"),
    ("model guidance", ["lora", "model", "checkpoint"], "models_loras"),
]


REQUIRED_FEATURE_KEYS = {
    "text_to_image",
    "image_prompt",
    "face_reference",
    "pyracanny",
    "cpds",
    "upscale",
    "variation",
    "inpaint",
    "auto_mask_sam",
    "auto_mask_u2net",
    "enhance",
    "describe",
    "styles",
    "models_loras",
}


def get_feature(key: str) -> StudioFeature:
    return FEATURES[key]


def list_features() -> List[StudioFeature]:
    return [FEATURES[key] for key in sorted(FEATURES)]


def validate_feature_map() -> None:
    missing = REQUIRED_FEATURE_KEYS - set(FEATURES)
    extra = set(FEATURES) - REQUIRED_FEATURE_KEYS
    if missing:
        raise AssertionError(f"Missing required Studio features: {sorted(missing)}")
    if extra:
        raise AssertionError(f"Unexpected Studio features: {sorted(extra)}")
    for key, feature in FEATURES.items():
        if feature.key != key:
            raise AssertionError(f"Feature key mismatch for {key}: {feature.key}")
        if not feature.label or not feature.fooocus_area:
            raise AssertionError(f"Feature {key} is missing label or Fooocus area")
        if not feature.use_when or not feature.avoid_when:
            raise AssertionError(f"Feature {key} is missing use/avoid guidance")
        if not feature.required_inputs or not feature.outputs or not feature.controls:
            raise AssertionError(f"Feature {key} is missing inputs, outputs, or controls")


def route_feature(goal: str, image_count: int, wants_identity: bool, wants_exact_edit: bool, wants_bundle: bool) -> StudioFeature:
    text = (goal or "").lower()
    if wants_bundle:
        return FEATURES["face_reference"]
    if wants_exact_edit:
        return FEATURES["inpaint"]
    for _scenario, keywords, feature_key in SCENARIO_RULES:
        if any(keyword in text for keyword in keywords):
            return FEATURES[feature_key]
    if image_count > 0 and wants_identity:
        return FEATURES["face_reference"]
    if image_count > 0:
        return FEATURES["image_prompt"]
    return FEATURES["text_to_image"]


def feature_map_markdown() -> str:
    validate_feature_map()
    lines = ["# AI Studio Feature Map", "", "This is the single source of truth used by the Studio Agent.", ""]
    for feature in list_features():
        lines.extend(
            [
                f"## {feature.label}",
                f"- Key: `{feature.key}`",
                f"- Fooocus area: {feature.fooocus_area}",
                f"- Use when: {feature.use_when}",
                f"- Avoid when: {feature.avoid_when}",
                f"- Required inputs: {', '.join(feature.required_inputs)}",
                f"- Outputs: {', '.join(feature.outputs)}",
                f"- Controls: {feature.controls}",
                f"- Notes: {feature.notes}",
                "",
            ]
        )
    return "\n".join(lines)
