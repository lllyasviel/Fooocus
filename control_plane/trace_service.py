from __future__ import annotations

from pathlib import Path

from .models import TraceResult


class TraceService:
    """Image trace/prompt-builder interface.

    First production adapter should call existing Fooocus interrogation code.
    Advanced adapters can later wrap Florence-2, Moondream, or LLaVA while
    returning the same TraceResult contract.
    """

    def trace_image(self, image_path: str, user_hint: str = "") -> TraceResult:
        path = Path(image_path)
        caption_parts = []
        if user_hint:
            caption_parts.append(user_hint.strip())
        caption_parts.append(f"reference image named {path.name}")
        caption = ", ".join(part for part in caption_parts if part)
        prompt = self.build_prompt(caption=caption, subjects=[], style_tags=[], lighting="", composition="")
        return TraceResult(
            image_path=str(path),
            caption=caption,
            suggested_prompt=prompt,
            suggested_negative_prompt="blurry, low quality, distorted, bad anatomy, watermark, text",
        )

    @staticmethod
    def build_prompt(
        caption: str,
        subjects: list[str],
        style_tags: list[str],
        lighting: str,
        composition: str,
    ) -> str:
        chunks = [caption]
        if subjects:
            chunks.append("subjects: " + ", ".join(subjects))
        if style_tags:
            chunks.append("style: " + ", ".join(style_tags))
        if lighting:
            chunks.append("lighting: " + lighting)
        if composition:
            chunks.append("composition: " + composition)
        chunks.append("high detail, clean composition, professional quality")
        return ", ".join(chunk for chunk in chunks if chunk)
