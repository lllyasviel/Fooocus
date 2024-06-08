from __future__ import annotations

from collections import UserList
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Literal, NamedTuple, Optional

try:
    from pydantic.v1 import (
        BaseModel,
        Extra,
        NonNegativeFloat,
        NonNegativeInt,
        PositiveInt,
        confloat,
        conint,
        validator,
    )
except ImportError:
    from pydantic import (
        BaseModel,
        Extra,
        NonNegativeFloat,
        NonNegativeInt,
        PositiveInt,
        confloat,
        conint,
        validator,
    )


@dataclass
class SkipImg2ImgOrig:
    steps: int
    sampler_name: str
    width: int
    height: int


class Arg(NamedTuple):
    attr: str
    name: str


class ArgsList(UserList):
    @cached_property
    def attrs(self) -> tuple[str, ...]:
        return tuple(attr for attr, _ in self)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self)


class ADetailerArgs(BaseModel, extra=Extra.forbid):
    ad_model: str = "None"
    ad_model_classes: str = ""
    ad_tap_enable: bool = True
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_confidence: confloat(ge=0.0, le=1.0) = 0.3
    ad_mask_k_largest: NonNegativeInt = 0
    ad_mask_min_ratio: confloat(ge=0.0, le=1.0) = 0.0
    ad_mask_max_ratio: confloat(ge=0.0, le=1.0) = 1.0
    ad_dilate_erode: int = 4
    ad_x_offset: int = 0
    ad_y_offset: int = 0
    ad_mask_merge_invert: Literal["None", "Merge", "Merge and Invert"] = "None"
    ad_mask_blur: NonNegativeInt = 4
    ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
    ad_inpaint_only_masked: bool = True
    ad_inpaint_only_masked_padding: NonNegativeInt = 32
    ad_use_inpaint_width_height: bool = False
    ad_inpaint_width: PositiveInt = 512
    ad_inpaint_height: PositiveInt = 512
    ad_use_steps: bool = False
    ad_steps: PositiveInt = 28
    ad_use_cfg_scale: bool = False
    ad_cfg_scale: NonNegativeFloat = 7.0
    ad_use_checkpoint: bool = False
    ad_checkpoint: Optional[str] = None
    ad_use_vae: bool = False
    ad_vae: Optional[str] = None
    ad_use_sampler: bool = False
    ad_sampler: str = "DPM++ 2M Karras"
    ad_scheduler: str = "Use same scheduler"
    ad_use_noise_multiplier: bool = False
    ad_noise_multiplier: confloat(ge=0.5, le=1.5) = 1.0
    ad_use_clip_skip: bool = False
    ad_clip_skip: conint(ge=1, le=12) = 1
    ad_restore_face: bool = False
    ad_controlnet_model: str = "None"
    ad_controlnet_module: str = "None"
    ad_controlnet_weight: confloat(ge=0.0, le=1.0) = 1.0
    ad_controlnet_guidance_start: confloat(ge=0.0, le=1.0) = 0.0
    ad_controlnet_guidance_end: confloat(ge=0.0, le=1.0) = 1.0
    is_api: bool = True

    @validator("is_api", pre=True)
    def is_api_validator(cls, v: Any):  # noqa: N805
        "tuple is json serializable but cannot be made with json deserialize."
        return type(v) is not tuple

    @staticmethod
    def ppop(
        p: dict[str, Any],
        key: str,
        pops: list[str] | None = None,
        cond: Any = None,
    ) -> None:
        if pops is None:
            pops = [key]
        if key not in p:
            return
        value = p[key]
        cond = (not bool(value)) if cond is None else value == cond

        if cond:
            for k in pops:
                p.pop(k, None)

    def extra_params(self, suffix: str = "") -> dict[str, Any]:
        if self.need_skip():
            return {}

        p = {name: getattr(self, attr) for attr, name in ALL_ARGS}
        ppop = partial(self.ppop, p)

        ppop("ADetailer model classes")
        ppop("ADetailer prompt")
        ppop("ADetailer negative prompt")
        p.pop("ADetailer tap enable", None)  # always pop
        ppop("ADetailer mask only top k largest", cond=0)
        ppop("ADetailer mask min ratio", cond=0.0)
        ppop("ADetailer mask max ratio", cond=1.0)
        ppop("ADetailer x offset", cond=0)
        ppop("ADetailer y offset", cond=0)
        ppop("ADetailer mask merge invert", cond="None")
        ppop("ADetailer inpaint only masked", ["ADetailer inpaint padding"])
        ppop(
            "ADetailer use inpaint width height",
            [
                "ADetailer use inpaint width height",
                "ADetailer inpaint width",
                "ADetailer inpaint height",
            ],
        )
        ppop(
            "ADetailer use separate steps",
            ["ADetailer use separate steps", "ADetailer steps"],
        )
        ppop(
            "ADetailer use separate CFG scale",
            ["ADetailer use separate CFG scale", "ADetailer CFG scale"],
        )
        ppop(
            "ADetailer use separate checkpoint",
            ["ADetailer use separate checkpoint", "ADetailer checkpoint"],
        )
        ppop(
            "ADetailer use separate VAE",
            ["ADetailer use separate VAE", "ADetailer VAE"],
        )
        ppop(
            "ADetailer use separate sampler",
            [
                "ADetailer use separate sampler",
                "ADetailer sampler",
                "ADetailer scheduler",
            ],
        )
        ppop("ADetailer scheduler", cond="Use same scheduler")
        ppop(
            "ADetailer use separate noise multiplier",
            ["ADetailer use separate noise multiplier", "ADetailer noise multiplier"],
        )

        ppop(
            "ADetailer use separate CLIP skip",
            ["ADetailer use separate CLIP skip", "ADetailer CLIP skip"],
        )

        ppop("ADetailer restore face")
        ppop(
            "ADetailer ControlNet model",
            [
                "ADetailer ControlNet model",
                "ADetailer ControlNet module",
                "ADetailer ControlNet weight",
                "ADetailer ControlNet guidance start",
                "ADetailer ControlNet guidance end",
            ],
            cond="None",
        )
        ppop("ADetailer ControlNet module", cond="None")
        ppop("ADetailer ControlNet weight", cond=1.0)
        ppop("ADetailer ControlNet guidance start", cond=0.0)
        ppop("ADetailer ControlNet guidance end", cond=1.0)

        if suffix:
            p = {k + suffix: v for k, v in p.items()}

        return p

    def is_mediapipe(self) -> bool:
        return self.ad_model.lower().startswith("mediapipe")

    def need_skip(self) -> bool:
        return self.ad_model == "None" or self.ad_tap_enable is False


_all_args = [
    ("ad_model", "ADetailer model"),
    ("ad_model_classes", "ADetailer model classes"),
    ("ad_tap_enable", "ADetailer tap enable"),
    ("ad_prompt", "ADetailer prompt"),
    ("ad_negative_prompt", "ADetailer negative prompt"),
    ("ad_confidence", "ADetailer confidence"),
    ("ad_mask_k_largest", "ADetailer mask only top k largest"),
    ("ad_mask_min_ratio", "ADetailer mask min ratio"),
    ("ad_mask_max_ratio", "ADetailer mask max ratio"),
    ("ad_x_offset", "ADetailer x offset"),
    ("ad_y_offset", "ADetailer y offset"),
    ("ad_dilate_erode", "ADetailer dilate erode"),
    ("ad_mask_merge_invert", "ADetailer mask merge invert"),
    ("ad_mask_blur", "ADetailer mask blur"),
    ("ad_denoising_strength", "ADetailer denoising strength"),
    ("ad_inpaint_only_masked", "ADetailer inpaint only masked"),
    ("ad_inpaint_only_masked_padding", "ADetailer inpaint padding"),
    ("ad_use_inpaint_width_height", "ADetailer use inpaint width height"),
    ("ad_inpaint_width", "ADetailer inpaint width"),
    ("ad_inpaint_height", "ADetailer inpaint height"),
    ("ad_use_steps", "ADetailer use separate steps"),
    ("ad_steps", "ADetailer steps"),
    ("ad_use_cfg_scale", "ADetailer use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer CFG scale"),
    ("ad_use_checkpoint", "ADetailer use separate checkpoint"),
    ("ad_checkpoint", "ADetailer checkpoint"),
    ("ad_use_vae", "ADetailer use separate VAE"),
    ("ad_vae", "ADetailer VAE"),
    ("ad_use_sampler", "ADetailer use separate sampler"),
    ("ad_sampler", "ADetailer sampler"),
    ("ad_scheduler", "ADetailer scheduler"),
    ("ad_use_noise_multiplier", "ADetailer use separate noise multiplier"),
    ("ad_noise_multiplier", "ADetailer noise multiplier"),
    ("ad_use_clip_skip", "ADetailer use separate CLIP skip"),
    ("ad_clip_skip", "ADetailer CLIP skip"),
    ("ad_restore_face", "ADetailer restore face"),
    ("ad_controlnet_model", "ADetailer ControlNet model"),
    ("ad_controlnet_module", "ADetailer ControlNet module"),
    ("ad_controlnet_weight", "ADetailer ControlNet weight"),
    ("ad_controlnet_guidance_start", "ADetailer ControlNet guidance start"),
    ("ad_controlnet_guidance_end", "ADetailer ControlNet guidance end"),
]

_args = [Arg(*args) for args in _all_args]
ALL_ARGS = ArgsList(_args)

BBOX_SORTBY = [
    "None",
    "Position (left to right)",
    "Position (center to edge)",
    "Area (large to small)",
]
MASK_MERGE_INVERT = ["None", "Merge", "Merge and Invert"]

_script_default = (
    "dynamic_prompting",
    "dynamic_thresholding",
    "wildcard_recursive",
    "wildcards",
    "lora_block_weight",
    "negpip",
)
SCRIPT_DEFAULT = ",".join(sorted(_script_default))

_builtin_script = ("soft_inpainting", "hypertile_script")
BUILTIN_SCRIPT = ",".join(sorted(_builtin_script))