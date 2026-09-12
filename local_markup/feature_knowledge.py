from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureKnowledge:
    key: str
    name: str
    area: str
    best_for: str
    avoid_for: str
    inputs: List[str]
    settings: List[str]
    notes: List[str]
    next_action: str


FEATURES: Dict[str, FeatureKnowledge] = {
    "text_to_image": FeatureKnowledge(
        "text_to_image",
        "Text to Image",
        "Prompt only",
        "brand-new images from a written idea",
        "precise edits to an existing image",
        ["main prompt", "negative prompt", "style choice"],
        ["Speed for drafts", "Quality for final", "2 to 4 candidates"],
        ["Use a specific subject, setting, lighting, and camera direction."],
        "Write one specific image prompt and generate candidates.",
    ),
    "image_prompt": FeatureKnowledge(
        "image_prompt",
        "Image Prompt Reference",
        "Image Prompt",
        "new images guided by uploaded references",
        "pixel-precise local edits",
        ["reference image", "main prompt", "negative prompt"],
        ["Use one clear reference first", "Add additional references only when they serve a purpose"],
        ["This creates a new image guided by the reference; it is not exact editing."],
        "Upload reference image and generate one shot at a time.",
    ),
    "face_reference": FeatureKnowledge(
        "face_reference",
        "Face Reference",
        "Image Prompt",
        "keeping a subject visually consistent across new outputs",
        "object-only or background-only edits",
        ["clear subject reference", "main prompt", "negative prompt"],
        ["Use a clear front or near-front reference", "Add pose/body reference separately when needed"],
        ["Use this when consistency matters more than loose style inspiration."],
        "Use the subject reference plus one specific shot prompt.",
    ),
    "pyracanny": FeatureKnowledge(
        "pyracanny",
        "PyraCanny Structure Control",
        "Image Prompt",
        "strong edge, outline, or pose guidance",
        "soft composition guidance",
        ["structure reference", "main prompt"],
        ["Use when the silhouette or edges matter", "Reduce influence if output is too stiff"],
        ["Stronger and more edge-driven than CPDS."],
        "Use a clean structure reference and generate controlled candidates.",
    ),
    "cpds": FeatureKnowledge(
        "cpds",
        "CPDS Composition Control",
        "Image Prompt",
        "natural composition and pose guidance",
        "exact edge copying",
        ["composition reference", "main prompt"],
        ["Use when PyraCanny is too rigid", "Pair with a subject reference when consistency matters"],
        ["Good for natural full-body or lifestyle layouts."],
        "Use CPDS for softer pose and composition guidance.",
    ),
    "upscale_variation": FeatureKnowledge(
        "upscale_variation",
        "Upscale / Variation",
        "Upscale or Variation",
        "making a good image larger, cleaner, or slightly varied",
        "major content replacement",
        ["source image"],
        ["Upscale 2x for winners", "Vary subtle for minor changes", "Vary strong for creative exploration"],
        ["Use after selecting a promising candidate."],
        "Use as a finishing or variation pass.",
    ),
    "inpaint": FeatureKnowledge(
        "inpaint",
        "Inpaint / Outpaint",
        "Inpaint or Outpaint",
        "changing only a selected part of an image",
        "creating a brand-new scene from scratch",
        ["source image", "mask", "inpaint prompt", "negative prompt"],
        ["Modify Content for replacement", "Improve Detail for small fixes", "Review mask before generating"],
        ["Mask quality controls the result more than prompt length."],
        "Create or review the mask, then generate 2 to 4 candidates.",
    ),
    "sam_mask": FeatureKnowledge(
        "sam_mask",
        "SAM Auto Mask",
        "Inpaint or Outpaint > Advanced Masking",
        "text-directed object masks",
        "broad subject/background separation",
        ["source image", "detection prompt", "reviewed mask"],
        ["Use simple detection words", "Target one thing at a time", "Review mask before generation"],
        ["Best for object or clothing-region selection by text."],
        "Generate mask from image, inspect it, then inpaint.",
    ),
    "u2net_mask": FeatureKnowledge(
        "u2net_mask",
        "U2Net Segmentation Mask",
        "Inpaint or Outpaint > Advanced Masking",
        "broad subject or background segmentation",
        "tiny object selection",
        ["source image", "segmentation model", "reviewed mask"],
        ["Use for broad subject/background separation", "Use SAM for named objects"],
        ["Better for broad segmentation than specific text-targeted masking."],
        "Use for broad separation, then refine if needed.",
    ),
    "enhance": FeatureKnowledge(
        "enhance",
        "Enhance",
        "Enhance",
        "polishing a selected winner",
        "first-draft exploration",
        ["candidate image", "enhance prompt"],
        ["Use after choosing a winner", "Avoid strong changes when consistency matters"],
        ["Enhance is a finishing pass."],
        "Enhance only the best candidate.",
    ),
    "describe": FeatureKnowledge(
        "describe",
        "Describe",
        "Describe",
        "turning an image into prompt language",
        "automatic prompt use without review",
        ["source image"],
        ["Review description before using it"],
        ["Useful for reverse-engineering style and composition."],
        "Generate description, review it, then adapt it.",
    ),
}


def all_features() -> List[FeatureKnowledge]:
    return list(FEATURES.values())


def get_feature(key: str) -> FeatureKnowledge:
    return FEATURES[key]


def rank_features(goal: str, image_count: int = 0, wants_identity: bool = True, wants_exact_edit: bool = False, wants_bundle: bool = False) -> List[FeatureKnowledge]:
    text = (goal or "").lower()
    scores = {key: 0 for key in FEATURES}

    keyword_map = {
        "text_to_image": ["new", "create", "brand", "idea"],
        "image_prompt": ["reference", "inspired", "style", "uploaded"],
        "face_reference": ["same", "consistent", "recognizable", "subject", "person"],
        "pyracanny": ["edge", "outline", "structure", "canny"],
        "cpds": ["composition", "pose", "full body", "standing", "depth"],
        "upscale_variation": ["upscale", "bigger", "cleaner", "variation", "sharper"],
        "inpaint": ["remove", "replace", "change", "edit", "only", "background"],
        "sam_mask": ["mask", "detect", "object", "shirt", "jacket", "glasses"],
        "u2net_mask": ["background", "subject", "person", "segmentation"],
        "enhance": ["enhance", "polish", "final", "detail"],
        "describe": ["describe", "analyze", "reverse", "prompt from image"],
    }

    for key, words in keyword_map.items():
        scores[key] += sum(2 for word in words if word in text)

    if image_count > 0:
        for key in ["image_prompt", "face_reference", "inpaint", "upscale_variation", "sam_mask", "u2net_mask", "cpds", "pyracanny"]:
            scores[key] += 1
    if wants_identity:
        for key in ["face_reference", "image_prompt", "cpds"]:
            scores[key] += 3
    if wants_exact_edit:
        for key in ["inpaint", "sam_mask", "u2net_mask"]:
            scores[key] += 4
    if wants_bundle:
        for key in ["face_reference", "image_prompt", "cpds", "upscale_variation"]:
            scores[key] += 4

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [FEATURES[key] for key in ranked_keys if scores[key] > 0]


def feature_matrix_markdown() -> str:
    rows = ["| Feature | Fooocus area | Best for | Avoid for |", "|---|---|---|---|"]
    for feature in all_features():
        rows.append(f"| {feature.name} | {feature.area} | {feature.best_for} | {feature.avoid_for} |")
    return "\n".join(rows)
