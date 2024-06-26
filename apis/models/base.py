"""Common models"""
from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)


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
    cn_weight: float | None = Field(default=0.5, ge=0, le=2, description="ControlNet weight")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip, description="ControlNet type")


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
