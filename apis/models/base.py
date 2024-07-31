"""Common models"""
from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)

from modules.async_worker import AsyncTask


class Lora(BaseModel):
    """Common params lora model"""
    enabled: bool
    model_name: str
    weight: float = Field(default=0.5, ge=-2, le=2)

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


class UpscaleOrVaryMethod(str, Enum):
    """Upscale or Vary method"""
    disable = "Disabled"
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'
    upscale_custom = 'Upscale (Custom)'


class OutpaintExpansion(str, Enum):
    """Outpaint expansion"""
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'


class ControlNetType(str, Enum):
    """ControlNet Type"""
    cn_ip = "ImagePrompt"
    cn_ip_face = "FaceSwap"
    cn_canny = "PyraCanny"
    cn_cpds = "CPDS"


class ImagePrompt(BaseModel):
    """Common params object ImagePrompt"""
    cn_img: str | None = Field(default=None, description="ControlNet image")
    cn_stop: float | None = Field(default=0.6, ge=0, le=1, description="ControlNet stop")
    cn_weight: float | None = Field(default=0.6, ge=0, le=2, description="ControlNet weight")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip, description="ControlNet type")


class MaskModel(str, Enum):
    """Inpaint mask model"""
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"
    isnet_anime = "isnet-anime"
    sam = "sam"


class EnhanceCtrlNets(BaseModel):
    enhance_enabled: bool = Field(default=False, description="Enable enhance control nets")
    enhance_mask_dino_prompt: str = Field(default="", description="Mask dino prompt")
    enhance_prompt: str = Field(default="", description="Prompt")
    enhance_negative_prompt: str = Field(default="", description="Negative prompt")
    enhance_mask_model: MaskModel = Field(default=MaskModel.sam, description="Mask model")
    enhance_mask_cloth_category: str = Field(default="full", description="Mask cloth category")
    enhance_mask_sam_model: str = Field(default="vit_b", description="one of vit_b vit_h vit_l")
    enhance_mask_text_threshold: float = Field(default=0.25, ge=0, le=1, description="Mask text threshold")
    enhance_mask_box_threshold: float = Field(default=0.3, ge=0, le=1, description="Mask box threshold")
    enhance_mask_sam_max_detections: int = Field(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all")
    enhance_inpaint_disable_initial_latent: bool = Field(default=False, description="Inpaint disable initial latent")
    enhance_inpaint_engine: str = Field(default="v2.6", description="Inpaint engine")
    enhance_inpaint_strength: float = Field(default=1, ge=0, le=1, description="Inpaint strength")
    enhance_inpaint_respective_field: float = Field(default=0.618, ge=0, le=1, description="Inpaint respective field")
    enhance_inpaint_erode_or_dilate: float = Field(default=0, ge=-64, le=64, description="Inpaint erode or dilate")
    enhance_mask_invert: bool = Field(default=False, description="Inpaint mask invert")


class GenerateMaskRequest(BaseModel):
    """
    generate mask request
    """
    image: str = Field(description="Image url or base64")
    mask_model: MaskModel = Field(default=MaskModel.isnet_general_use, description="Mask model")
    cloth_category: str = Field(default="full", description="Mask cloth category")
    dino_prompt_text: str = Field(default="", description="Detection prompt, Use singular whenever possible")
    sam_model: str = Field(default="vit_b", description="one of vit_b vit_h vit_l")
    box_threshold: float = Field(default=0.3, ge=0, le=1, description="Mask box threshold")
    text_threshold: float = Field(default=0.25, ge=0, le=1, description="Mask text threshold")
    sam_max_detections: int = Field(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all")
    dino_erode_or_dilate: float = Field(default=0, ge=-64, le=64, description="Mask dino erode or dilate")
    dino_debug: bool = Field(default=False, description="Mask dino debug")


class DescribeImageType(str, Enum):
    """Image type for image to prompt"""
    photo = 'Photo'
    anime = 'Anime'


class DescribeImageResponse(BaseModel):
    """
    describe image response
    """
    describe: str


class CurrentTask:
    """
    Current task class.
    """
    ct = None
    task: AsyncTask = None
