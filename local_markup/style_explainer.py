from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from modules.sdxl_styles import apply_style, legal_style_names, styles


STYLE_EXPECTATIONS: Dict[str, str] = {
    "Fooocus V2": "Adds Fooocus prompt expansion. Expect richer detail, more descriptive composition, and stronger SDXL interpretation.",
    "Fooocus Photograph": "Pushes photographic realism. Good for portraits, headshots, products, and real-world scenes.",
    "Fooocus Enhance": "Improves polish and detail. Good when the image is close but needs a cleaner finish.",
    "Fooocus Sharp": "Adds crispness and edge detail. Good for headshots, product photos, and final presentation images.",
    "Fooocus Cinematic": "Adds dramatic lighting and movie-like contrast. Use when realism with mood matters.",
    "Fooocus Semi Realistic": "Blends realistic and stylized output. Use when you want less strict photo realism.",
    "Fooocus Negative": "Adds common negative constraints. Usually useful, but can over-constrain if stacked with too many negatives.",
    "Random Style": "Picks a random style. Good for exploration, not good for precise edits.",
}


@dataclass
class StylePreview:
    name: str
    expectation: str
    positive_preview: str
    negative_preview: str
    has_prompt_slot: bool

    def as_markdown(self) -> str:
        return (
            f"### {self.name}\n"
            f"**Expectation:** {self.expectation}\n\n"
            f"**Positive style effect:**\n```txt\n{self.positive_preview}\n```\n\n"
            f"**Negative style effect:**\n```txt\n{self.negative_preview or 'No style-specific negative prompt.'}\n```\n\n"
            f"**Prompt slot:** {'Uses your prompt inside the style template.' if self.has_prompt_slot else 'Appends or expands around your prompt.'}"
        )


def _clean_preview(lines: List[str], limit: int = 700) -> str:
    text = ", ".join([line.strip() for line in lines if line and line.strip()])
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def describe_style(style_name: str, sample_prompt: str = "professional realistic headshot") -> StylePreview:
    style_name = style_name or "Fooocus Photograph"
    sample_prompt = sample_prompt or "professional realistic headshot"

    if style_name == "Fooocus V2":
        return StylePreview(
            name=style_name,
            expectation=STYLE_EXPECTATIONS[style_name],
            positive_preview="Fooocus V2 expands the prompt dynamically at generation time.",
            negative_preview="No fixed negative prompt from this style.",
            has_prompt_slot=True,
        )

    if style_name == "Random Style":
        return StylePreview(
            name=style_name,
            expectation=STYLE_EXPECTATIONS[style_name],
            positive_preview="A random style will be selected at generation time.",
            negative_preview="Depends on the random style selected.",
            has_prompt_slot=False,
        )

    if style_name not in styles:
        style_name = "Fooocus Photograph" if "Fooocus Photograph" in styles else next(iter(styles.keys()))

    positive_lines, negative_lines, has_prompt_slot = apply_style(style_name, sample_prompt)
    return StylePreview(
        name=style_name,
        expectation=STYLE_EXPECTATIONS.get(
            style_name,
            "Applies this style template to the prompt. Expect the visual mood, medium, lighting, and rendering language to shift according to the style text.",
        ),
        positive_preview=_clean_preview(positive_lines),
        negative_preview=_clean_preview(negative_lines),
        has_prompt_slot=has_prompt_slot,
    )


def list_style_names() -> List[str]:
    return legal_style_names


def describe_style_markdown(style_name: str, sample_prompt: str = "professional realistic headshot") -> str:
    return describe_style(style_name, sample_prompt).as_markdown()


def style_recommendations_for_goal(goal: str) -> str:
    text = (goal or "").lower()
    if any(word in text for word in ["headshot", "portrait", "face", "person", "professional"]):
        recommended = ["Fooocus Photograph", "Fooocus Enhance", "Fooocus Sharp"]
        avoid = ["Random Style", "SAI Anime", "SAI Comic Book", "SAI 3D Model"]
    elif any(word in text for word in ["cinematic", "dramatic", "movie"]):
        recommended = ["Fooocus Cinematic", "MRE Cinematic Dynamic", "Fooocus Photograph"]
        avoid = ["Random Style"]
    else:
        recommended = ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
        avoid = ["Random Style for exact edits"]

    return (
        "Recommended styles:\n"
        + "\n".join(f"- {name}" for name in recommended)
        + "\n\nAvoid for this goal:\n"
        + "\n".join(f"- {name}" for name in avoid)
    )
