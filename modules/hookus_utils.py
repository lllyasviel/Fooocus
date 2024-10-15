import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
running_dir = os.path.dirname(current_dir)
sys.path.append(running_dir)
# This is a workaround to import modules from the parent directory

import numpy
import random
from pydantic import BaseModel, field_validator
from typing import Literal, Optional

from modules import config, constants, flags
from modules.sdxl_styles import legal_style_names
import modules.style_sorter as style_sorter

style_sorter.try_load_sorted_styles(legal_style_names, config.default_styles)
config.update_files()

all_styles = style_sorter.all_styles
performance_lora_keys = flags.PerformanceLoRA.__members__.keys()
performance_keys = flags.Performance.__members__.keys()

ALLOWED_TABS = Literal['uov', 'inpaint', 'ip', 'desc', 'enhance', 'metadata']
OUTPAINT_SELECTIONS = Literal['Left', 'Right', 'Top', 'Bottom']
REFINER_SWAP_METHODS = Literal['joint', 'separate', 'vae']

DEFAULT_PERFORMANCE_SELECTION = flags.Performance.HYPER_SD

lora_ctrls_pre = [LoraTuple(enabled=enbled, model=model, weight=weight) for enbled, model, weight in config.default_loras]
lora_ctrls = []
for lora in lora_ctrls_pre:
    lora_ctrls.append(lora.to_array())

class LoraTuple(BaseModel):
    enabled: bool = False
    model: str = ""
    weight: float = 1.0

    @field_validator("weight", mode="after")
    def validate_lora_weight(cls, v):
        if v < config.default_loras_min_weight or v > config.default_loras_max_weight:
            raise ValueError(f"Invalid lora_weight value: {v}")
        return v
    
    def to_array(self):
        return [self.enabled, self.model, self.weight]


class ControlNetImageTask(BaseModel):
    cn_img: Optional[numpy.ndarray] = None
    cn_stop: Optional[float] = None
    cn_weight: Optional[float] = 1.0
    cn_type: Optional[str] = flags.default_ip

    class Config:
        arbitrary_types_allowed = True

initial_cn_tasks = {x: [] for x in flags.ip_list}
for initial_cn_task in initial_cn_tasks: # Hoping that the controlnet image count is 4
    initial_cn_tasks[initial_cn_task] = ControlNetImageTask(cn_img=None, cn_stop=flags.default_parameters[initial_cn_task][0], cn_weight=flags.default_parameters[initial_cn_task][1])

class EnhanceMaskCtrls(BaseModel):
    """
    Enhacement mask controls for inpaint and outpaint
    
    """
    enhance_mask_dino_prompt_text: str = config.example_enhance_detection_prompts[0]
    enhance_prompt: str = None # Enhacement positive prompt. Uses original prompt if None
    enhance_negative_prompt: str = None # Enhacement negative prompt. Uses original negative prompt if None
    enhance_mask_model: str = config.default_enhance_inpaint_mask_model
    enhance_mask_cloth_category: str = config.default_inpaint_mask_cloth_category
    
    enhance_mask_sam_model: str = config.default_inpaint_mask_sam_model
    
    enhance_mask_text_threshold: float = 0.25 # min 0.0 max 1.0
    enhance_mask_box_threshold: float = 0.30 # min 0.0 max 1.0
    enhance_mask_sam_max_detections: int = config.default_sam_max_detections # min 1 max 10, set 0 to detect all
    enhance_inpaint_disable_initial_latent: bool = False
    enhance_inpaint_engine: str = config.default_inpaint_engine_version
        
    enhance_inpaint_strength: float = 1.0 # min 0.0 max 1.0
    enhance_inpaint_respective_field: float = 0.618 # min 0.0 max 1.0
    enhance_inpaint_erode_or_dilate: int = 0 # min -64 max 64
    enhance_mask_inver: bool = False

    @field_validator("enhance_mask_model", mode="after")
    def validate_enhance_mask_model(cls, v):
        if v not in flags.inpaint_mask_models:
            raise ValueError(f"Invalid inpaint mask model: {v}")
        return v 
    
    @field_validator("enhance_inpaint_engine", mode="after")
    def validate_enhance_inpaint_engine(cls, v):
        if v not in flags.inpaint_engine_versions:
            raise ValueError(f"Invalid inpaint engine version: {v}")
        return v
    
    @field_validator("enhance_mask_sam_model", mode="after")
    def validate_enhance_mask_sam_model(cls, v):
        if v not in flags.inpaint_mask_sam_model:
            raise ValueError(f"Invalid inpaint mask sam model: {v}")
        return v
    
    @field_validator("enhance_mask_cloth_category", mode="after")
    def validate_enhance_mask_cloth_category(cls, v):
        if v not in flags.inpaint_mask_cloth_category:
            raise ValueError(f"Invalid inpaint mask cloth category: {v}")
        return v
    


class PdAcyncTask(BaseModel):

    class Config:
        arbitrary_types_allowed = True


    yields: list = []
    results: list = []
    last_stop: bool = False
    processing: bool = True
    generate_image_grid: bool = False
    prompt: str = "A funny cat"
    negative_prompt: str = ""
    style_selections: list[str] = config.default_styles
    performance_selection: str = DEFAULT_PERFORMANCE_SELECTION
    performance_loras: list = []
    original_steps: int = -1
    steps: int = -1
    aspect_ratios_selection: str = config.default_aspect_ratio
    image_number: int = 1
    output_format: str = config.default_output_format
    
    seed: int = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    read_wildcards_in_order: bool = False # Read wildcards in order
    sharpness: float = config.default_sample_sharpness
    cfg_scale: float = config.default_cfg_scale # Aka guidance scale
    base_model_name: str = config.default_base_model_name 
    refiner_model_name: str = config.default_refiner_model_name
    refiner_switch: bool = config.default_refiner_switch
    loras: list = lora_ctrls

    input_image_checkbox: bool = False
    current_tab: ALLOWED_TABS = "uov" # upscale or variation
    uov_method: str = config.default_enhance_uov_method
    uov_input_image: numpy.ndarray = None # TODO: trigger_auto_describe
    outpaint_selections: Optional[OUTPAINT_SELECTIONS | list] = []
    inpaint_input_image: numpy.ndarray = None # TODO: trigger_auto_describe
    inpaint_additional_prompt: str = None
    inpaint_mask_image_upload: numpy.ndarray = None

    disable_preview: bool = False
    disable_intermediate_results: bool = False
    disable_seed_increment: bool = False
    black_out_nsfw: bool = config.default_black_out_nsfw

    # TODO type checks
    adm_scaler_positive: float = 1.5 # Min 0.1 max 3.0
    adm_scaler_negative: float = 0.8 # Min 0.1 max 3.0
    adm_scaler_end: float = 0.3 # Min 0.0 max 1.0
    
    adaptive_cfg: float = config.default_cfg_tsnr # min 1.0 max 30.0
    clip_skip: int = config.default_clip_skip # min 1 max config.clip_skip_max
    sampler_name: str = config.default_sampler
    scheduler_name: str = config.default_scheduler
    vae_name: str = config.default_vae
    overwrite_step: int = config.default_overwrite_step
    overwrite_switch: int = config.default_overwrite_switch
    overwrite_width: int = -1
    overwrite_height: int = -1
    overwrite_vary_strength: float = -1
    overwrite_upscale_strength: float = config.default_overwrite_upscale

    
    mixing_image_prompt_and_vary_upscale: bool = False
    mixing_image_prompt_and_inpaint: bool = False
    debugging_cn_preprocessor: bool = False
    skipping_cn_preprocessor: bool = False
    canny_low_threshold: int = 64 # min 0 max 255

    canny_high_threshold: int = 128 # min 0 max 255
    refiner_swap_method: REFINER_SWAP_METHODS = flags.refiner_swap_method
    controlnet_softness: float = 0.25 # min 0.0 max 1.0
    freeu_enabled: bool = False
    freeu_b1: float = 1.01 # min 0.0 max 2.0
    freeu_b2: float = 1.02 # min 0.0 max 2.0
    freeu_s1: float = 0.99 # min 0.0 max 2.0
    freeu_s2: float = 0.95 # min 0.0 max 2.0

    debugging_inpaint_preprocessor: bool = False
    inpaint_disable_initial_latent: bool = False
    inpaint_engine: str = config.default_inpaint_engine_version
    @field_validator("inpaint_engine", mode="after")
    def validate_inpaint_engine(cls, v):
        if v not in flags.inpaint_engine_versions:
            raise ValueError(f"Invalid inpaint engine version: {v}")
        return v
        
    


    inpaint_strength: float = 1.0 # min 0.0 max 1.0
    inpaint_respective_field: float = 0.618 # min 0.0 max 1.0
    inpaint_advanced_masking_checkbox: bool = False
    invert_mask_checkbox: bool = False
    inpaint_erode_or_dilate: int = 0 # min -64 max 64
    save_metadata_to_images: bool = config.default_save_metadata_to_images
    
    args_disable_metadata: bool = True
    metadata_scheme: str = config.default_metadata_scheme

    cn_tasks: dict = initial_cn_tasks # TODO this will need to be parsed back into an array for the async worker...

    debugging_dino: bool = False
    dino_erode_or_dilate: int = 0 # min -64 max 64
    debugging_enhance_masks_checkbox: bool = False

    enhance_input_image: numpy.ndarray = None
    enhance_checkbox: bool = config.default_enhance_checkbox
    enhance_uov_method: str = config.default_enhance_uov_method
    enhance_uov_processing_order: str = config.default_enhance_uov_processing_order

    enhance_uov_prompt_type: str = config.default_enhance_uov_prompt_type
    enhance_ctrls: Optional[list[EnhanceMaskCtrls]] = []

    @field_validator("output_format")
    def validate_output_format(cls, v):
        if v not in flags.output_formats:
            raise ValueError(f"Invalid output format: {v}")
        return v
    
    @field_validator("aspect_ratios_selection")
    def validate_aspect_ratios_selection(cls, v):
        if v not in config.available_aspect_ratios:
            raise ValueError(f"Invalid aspect ratio selection: {v}")
        return v
    
    @field_validator("performance_selection", mode="after")
    def validate_performance_selection(cls, v):
        if v not in flags.Performance.values():
            raise ValueError(f"Invalid performance selection: {v}")
        return v
    
    @field_validator("seed", mode="after")
    def validate_seed(cls, v):
        if v < constants.MIN_SEED or v > constants.MAX_SEED:
            raise ValueError(f"Invalid seed value: {v}")
        return v
    
    @field_validator("style_selections")
    def validate_style_selections(cls, v):
        for style in v:
            if style not in all_styles:
                raise ValueError(f"Invalid style selection: {style}")
        return v
    
    @field_validator("sharpness", mode="after")
    def validate_sharpness(cls, v):
        if v < 0.0 or v > 30.0:
            raise ValueError(f"Invalid sharpness value: {v}. The value must be between 0.0 and 30.0")
        return v
    
    @field_validator("cfg_scale", mode="after")
    def validate_cfg_scale(cls, v):
        if v < 1.0 or v > 30.0:
            raise ValueError(f"Invalid cfg_scale value: {v}. The value must be between 1.0 and 30.0")
        return v
    
    @field_validator("refiner_switch", mode="after")
    def validate_refiner_switch(cls, v):
        if v < 0.1 or v > 1.0:
            raise ValueError(f"Invalid refiner_switch value: {v}. The value must be either 0 or 1")
        
    @field_validator("sampler_name")
    def validate_sampler_name(cls, v):
        if v not in flags.sampler_list:
            raise ValueError(f"Invalid sampler name: {v}")
        return v
    
    @field_validator("scheduler_name")
    def validate_scheduler_name(cls, v):
        if v not in flags.scheduler_list:
            raise ValueError(f"Invalid scheduler name: {v}")
        return v

    @field_validator("overwrite_step")
    def validate_overwrite_step(cls, v):
        # -1 to disable
        if v < -1 or v > 200:
            raise ValueError(f"Invalid overwrite step: {v}")
        return v
    
    @field_validator("overwrite_vary_strength")
    def validate_overwrite_vary_strength(cls, v):
        # -1 to disable
        if v < -1 or v > 1.0:
            raise ValueError(f"Invalid overwrite vary strength: {v}")
        return v
    
    @field_validator("vae_name")
    def validate_vae_name(cls, v):
        if v not in flags.default_vae + config.vae_filenames:
            raise ValueError(f"Invalid vae name: {v}")
        return v
    
    @field_validator("overwrite_height")
    def validate_overwrite_height(cls, v):
        # -1 to disable
        if v < -1 or v > 2048:
            raise ValueError(f"Invalid overwrite height: {v}")
        return v

    @field_validator("overwrite_width")
    def validate_overwrite_width(cls, v):
        # -1 to disable
        if v < -1 or v > 2048:
            raise ValueError(f"Invalid overwrite width: {v}")
        return v

    @field_validator("overwrite_upscale_strength")
    def validate_overwrite_upscale_strength(cls, v):
        # -1 to disable
        if v < -1 or v > 1.0:
            raise ValueError(f"Invalid overwrite upscale strength: {v}")
        return v
    
    @field_validator("overwrite_switch")
    def validate_overwrite_switch(cls, v):
        # -1 to disable
        if v < -1 or v > 200:
            raise ValueError(f"Invalid overwrite switch: {v}")
        
    @field_validator("enhance_uov_method", mode="after")
    def validate_enhance_uov_method(cls, v):
        if v not in flags.uov_list:
            raise ValueError(f"Invalid enhance uov method: {v}")
        return v
    
    @field_validator("enhance_uov_processing_order", mode="after")
    def validate_enhance_uov_processing_order(cls, v):
        if v not in config.default_enhance_uov_processing_order:
            raise ValueError(f"Invalid enhance uov processing order: {v}")
        return v
    
    @field_validator("enhance_uov_prompt_type", mode="after")
    def validate_enhance_uov_prompt_type(cls, v):
        if v not in flags.enhancement_uov_prompt_types:
            raise ValueError(f"Invalid enhance uov prompt type: {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        perf_name = flags.Performance(self.performance_selection).name
        perf_lora = flags.PerformanceLoRA[perf_name].value
        if perf_lora:
            self.performance_loras.append(perf_lora)
        self.steps = self.steps if self.steps != -1 else flags.Steps[perf_name].value
        self.original_steps = self.original_steps if self.original_steps != -1 else self.steps


if __name__ == "__main__":
    test_obj = PdAcyncTask()
    ...
    


