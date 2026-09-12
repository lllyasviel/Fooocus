from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


SAFE_OUTFIT_MAP: Dict[str, str] = {
    "swimwear": "wearing tasteful swim trunks suitable for swimming, beach-ready, non-explicit, natural fit",
    "resort": "wearing a clean resort casual outfit, short sleeve shirt, relaxed vacation style",
    "fitness": "wearing athletic training clothes, gym-ready, clean modern fit",
    "business": "wearing a professional business outfit, polished executive look",
    "casual": "wearing clean casual clothes, modern everyday style",
}


BUNDLE_PRESETS: Dict[str, List[str]] = {
    "Lifestyle Starter": [
        "professional headshot",
        "standing full-body casual portrait",
        "outdoor lifestyle photo",
        "studio portrait",
    ],
    "Swimming / Resort": [
        "standing by a pool in tasteful swimwear",
        "walking near the beach in resort light",
        "relaxed resort portrait",
        "sunny vacation profile photo",
    ],
    "Executive + Casual": [
        "executive headshot",
        "business casual standing portrait",
        "smart casual lifestyle photo",
        "confident outdoor portrait",
    ],
    "Fitness / Transformation": [
        "athletic standing portrait",
        "gym lifestyle photo",
        "outdoor training look",
        "clean fitness profile portrait",
    ],
}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def allow_swimwear() -> bool:
    return env_flag("KJB_ALLOW_SWIMWEAR_BUNDLES", True)


def strict_photo_guardrails() -> bool:
    return env_flag("KJB_STRICT_PHOTO_GUARDRAILS", True)


@dataclass
class PhotoBundlePlan:
    preset: str
    outfit: str
    identity_prompt: str
    negative_prompt: str
    shots: List[str]
    guidance: str
    safety_status: str

    def as_text(self) -> str:
        lines = [
            f"Bundle: {self.preset}",
            f"Outfit: {self.outfit}",
            "",
            "Safety status:",
            self.safety_status,
            "",
            "Identity prompt:",
            self.identity_prompt,
            "",
            "Negative prompt:",
            self.negative_prompt,
            "",
            "Shots:",
        ]
        lines.extend([f"{index + 1}. {shot}" for index, shot in enumerate(self.shots)])
        lines.extend(["", "Guidance:", self.guidance])
        return "\n".join(lines)


def build_identity_prompt(user_goal: str, outfit: str) -> Tuple[str, str, str]:
    if outfit == "swimwear" and not allow_swimwear():
        outfit = "resort"
        safety_status = "Swimwear bundles are disabled by KJB_ALLOW_SWIMWEAR_BUNDLES=false; using resort outfit instead."
    elif strict_photo_guardrails():
        safety_status = "Strict photo guardrails enabled. The bundle is limited to safe outfit, swimwear, resort, fitness, business, and casual looks."
    else:
        safety_status = "Strict photo guardrails disabled for styling freedom, but the workflow still uses safe non-explicit outfit directions."

    outfit_text = SAFE_OUTFIT_MAP.get(outfit, SAFE_OUTFIT_MAP["casual"])
    goal = (user_goal or "").strip() or "create a realistic personal photo set"
    prompt = (
        "same adult person from the reference photos, preserve facial identity, age, skin tone, face shape, natural expression, "
        "realistic body proportions, natural posture, realistic hands, consistent lighting, high quality photography, "
        f"{outfit_text}, goal: {goal}"
    )
    return prompt, outfit, safety_status


def build_bundle_plan(preset: str, outfit: str, user_goal: str) -> PhotoBundlePlan:
    preset = preset if preset in BUNDLE_PRESETS else "Lifestyle Starter"
    outfit = outfit if outfit in SAFE_OUTFIT_MAP else "casual"
    identity_prompt, final_outfit, safety_status = build_identity_prompt(user_goal, outfit)
    shots = [f"{identity_prompt}, {shot}" for shot in BUNDLE_PRESETS[preset]]
    negative_prompt = (
        "different person, changed identity, unsafe adult content, see-through clothing, sexualized pose, deformed face, "
        "distorted body, extra limbs, bad hands, warped fingers, blurry, low quality, unrealistic anatomy, melted clothing"
    )
    guidance = (
        "Use 2-5 clear reference photos when possible: one face close-up, one upper body, one full body if available, and one natural expression. "
        "A headshot alone can inspire a standing image, but full-body accuracy is better with a full-body reference. "
        "For outfit changes, use tasteful clothing or swimwear only. "
        "Environment controls: KJB_ALLOW_SWIMWEAR_BUNDLES=true/false and KJB_STRICT_PHOTO_GUARDRAILS=true/false."
    )
    return PhotoBundlePlan(preset, final_outfit, identity_prompt, negative_prompt, shots, guidance, safety_status)


def build_bundle_outputs(preset: str, outfit: str, user_goal: str) -> Tuple[str, str, str]:
    plan = build_bundle_plan(preset, outfit, user_goal)
    return plan.identity_prompt, plan.negative_prompt, plan.as_text()
