"""Common model for requests"""
from typing import List
from pydantic import (
    BaseModel,
    Field,
    StrictStr
)
from apis.models.base import (
    DescribeImageType, EnhanceCtrlNets, ImagePrompt,
    Lora,
    UpscaleOrVaryMethod,
    OutpaintExpansion
)

from modules.config import (
    default_prompt,
    default_prompt_negative,
    default_styles,
    default_output_format,
    default_aspect_ratio,
    default_sample_sharpness,
    default_cfg_scale,
    default_base_model_name,
    default_black_out_nsfw,
    default_refiner_model_name,
    default_refiner_switch,
    default_loras,
    default_cfg_tsnr,
    default_clip_skip,
    default_sampler,
    default_scheduler,
    default_vae,
    default_overwrite_step,
    default_overwrite_switch,
    default_inpaint_engine_version
)
from modules.flags import (
    clip_skip_max,
    refiner_swap_method,
    Performance,
    MetadataScheme
)

default_aspect_ratio = default_aspect_ratio.split(" ")[0].replace("×", "*")

loras = []
for lora in default_loras:
    loras.append(Lora(
        enabled=lora[0],
        model_name=lora[1],
        weight=lora[2]
    ))


class CommonRequest(BaseModel):
    """All generate request based on this model"""
    prompt: StrictStr = Field(default=default_prompt, description="Prompt to generate image")
    negative_prompt: StrictStr = Field(default=default_prompt_negative, description="Negative prompt to filter out unwanted content")
    style_selections: List[StrictStr] = Field(default=default_styles, description="Style to generate image")
    performance_selection: Performance = Field(default=Performance.SPEED, description="Performance selection")
    aspect_ratios_selection: StrictStr = Field(default=default_aspect_ratio, description="Aspect ratio selection")
    image_number: int = Field(default=1, description="Image number", ge=1, le=32)
    output_format: StrictStr = Field(default=default_output_format, description="Output format")
    image_seed: int = Field(default=-1, description="Seed to generate image, -1 for random")
    read_wildcards_in_order: bool = Field(default=False, description="Read wildcards in order")
    sharpness: float = Field(default=default_sample_sharpness, ge=0.0, le=30.0)
    guidance_scale: float = Field(default=default_cfg_scale, ge=1.0, le=30.0)
    base_model_name: StrictStr = Field(default=default_base_model_name, description="Base Model Name")
    refiner_model_name: StrictStr = Field(default_refiner_model_name, description="Refiner Model Name")
    refiner_switch: float = Field(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0)
    loras: List[Lora] = Field(default=loras, description="Lora")

    input_image_checkbox: bool = Field(default=False, description="Input Image")
    current_tab: StrictStr = Field(default='uov', description="Current tab")
    uov_method: UpscaleOrVaryMethod = Field(default=UpscaleOrVaryMethod.disable, description="Upscale or Vary Method")
    uov_input_image: StrictStr | None = Field(default="None", description="Upscale or Vary Input Image")
    outpaint_selections: List[OutpaintExpansion] = Field(default=[], description="Outpaint Expansion")
    inpaint_input_image: StrictStr | None = Field(default="None", description="Inpaint Input Image")
    inpaint_additional_prompt: StrictStr = Field(default="", description="Additional prompt for inpaint")
    inpaint_mask_image_upload: str | None = Field(default="None", description="Inpaint Mask Image Upload")
    disable_preview: bool = Field(default=False, description="Disable preview")
    disable_intermediate_results: bool = Field(default=False, description="Disable intermediate results")
    disable_seed_increment: bool = Field(default=False, description="Disable seed increment")
    black_out_nsfw: bool = Field(default=default_black_out_nsfw, description="Black out NSFW")
    adm_scaler_positive: float = Field(default=1.5, ge=0.0, le=3.0, description="The scaler multiplied to positive ADM (use 1.0 to disable).")
    adm_scaler_negative: float = Field(default=0.8, ge=0.0, le=3.0, description="The scaler multiplied to negative ADM (use 1.0 to disable).")
    adm_scaler_end: float = Field(default=0.3, ge=0.0, le=1.0, description="ADM Guidance End At Step")
    adaptive_cfg: float = Field(default=default_cfg_tsnr, ge=1.0, le=30.0, description="Adaptive cfg")
    clip_skip: int = Field(default=default_clip_skip, ge=1, le=clip_skip_max, description="Clip skip")
    sampler_name: StrictStr = Field(default=default_sampler, description="Sampler name")
    scheduler_name: StrictStr = Field(default=default_scheduler, description="Scheduler name")
    vae_name: StrictStr = Field(default=default_vae, description="VAE name")
    overwrite_step: int = Field(default=default_overwrite_step, description="Overwrite step")
    overwrite_switch: int = Field(default=default_overwrite_switch, description="Overwrite switch")
    overwrite_width: int = Field(default=-1, ge=-1, le=2048, description="Overwrite width")
    overwrite_height: int = Field(default=-1, ge=-1, le=2048, description="Overwrite height")
    overwrite_vary_strength: float = Field(default=-1, ge=-1, le=1.0, description="Overwrite vary strength")
    overwrite_upscale_strength: float = Field(default=-1, ge=-1, le=1.0, description="Overwrite upscale strength")
    mixing_image_prompt_and_vary_upscale: bool = Field(default=False, description="Mixing image prompt and vary upscale")
    mixing_image_prompt_and_inpaint: bool = Field(default=False, description="Mixing image prompt and inpaint")
    debugging_cn_preprocessor: bool = Field(default=False, description="Debugging cn preprocessor")
    skipping_cn_preprocessor: bool = Field(default=False, description="Skipping cn preprocessor")
    canny_low_threshold: int = Field(default=64, ge=1, le=255, description="Canny Low Threshold")
    canny_high_threshold: int = Field(default=128, ge=1, le=255, description="Canny High Threshold")
    refiner_swap_method: StrictStr = Field(default=refiner_swap_method, description="Refiner Swap Method")
    controlnet_softness: float = Field(default=0.25, ge=0.0, le=1.0, description="ControlNet Softness")
    freeu_enabled: bool = Field(default=False, description="Enable freeu")
    freeu_b1: float = Field(default=1.01, description="Freeu B1")
    freeu_b2: float = Field(default=1.02, description="Freeu B2")
    freeu_s1: float = Field(default=0.99, description="Freeu S1")
    freeu_s2: float = Field(default=0.95, description="Freeu S2")
    debugging_inpaint_preprocessor: bool = Field(default=False, description="Debugging inpaint preprocessor")
    inpaint_disable_initial_latent: bool = Field(default=False, description="Disable initial latent")
    inpaint_engine: StrictStr = Field(default=default_inpaint_engine_version, description="Inpaint Engine")
    inpaint_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Inpaint Denoising Strength")
    inpaint_respective_field: float = Field(default=0.618, ge=0.0, le=1.0,
                                            description="""
                                            Inpaint Respective Field
                                            The area to inpaint.
                                            Value 0 is same as "Only Masked" in A1111.
                                            Value 1 is same as "Whole Image" in A1111.
                                            Only used in inpaint, not used in outpaint.
                                            (Outpaint always use 1.0)
                                            """)
    # inpaint_mask_upload_checkbox: bool = Field(default=False, description="Inpaint Mask Upload Checkbox")
    inpaint_advanced_masking_checkbox: bool = Field(default=False, description="Inpaint Advanced Masking Checkbox")
    invert_mask_checkbox: bool = Field(default=False, description="Inpaint Invert Mask Checkbox")
    inpaint_erode_or_dilate: int = Field(default=0, ge=-64, le=64, description="Inpaint Erode or Dilate")
    save_final_enhanced_image_only: bool = Field(default=False, description="Save final enhanced image only")
    save_metadata_to_images: bool = Field(default=True, description="Save meta data")
    metadata_scheme: MetadataScheme = Field(default=MetadataScheme.FOOOCUS, description="Meta data scheme, one of [fooocus, a111]")
    controlnet_image: List[ImagePrompt] = Field(default=[ImagePrompt()], description="ControlNet Image Prompt")
    debugging_dino: bool = Field(default=False, description="Debugging DINO")
    dino_erode_or_dilate: int = Field(default=0, ge=-64, le=64, description="DINO Erode or Dilate")
    debugging_enhance_masks_checkbox: bool = Field(default=False, description="Debugging Enhance Masks")
    enhance_input_image: str | None = Field(default="None", description="Enhance Input Image")
    enhance_checkbox: bool = Field(default=False, description="Enhance Checkbox")
    enhance_uov_method: UpscaleOrVaryMethod = Field(default=UpscaleOrVaryMethod.disable, description="Upscale or Vary Method")
    enhance_uov_processing_order: str = Field(default='Before First Enhancement', description="Enhance UOV Processing Order, one of [Before First Enhancement, After Last Enhancement]")
    enhance_uov_prompt_type: str = Field(default='Original Prompts', description="One of 'Last Filled Enhancement Prompts', 'Original Prompts', work with enhance_uov_processing_order='After Last Enhancement'")
    enhance_ctrls: List[EnhanceCtrlNets] = Field(default=[], description="Enhance Control Nets")

    generate_image_grid: bool = Field(default=False, description="Generate Image Grid for Each Batch, (Experimental) This may cause performance problems on some computers and certain internet conditions.")

    save_name: str = Field(default=None, description="You can diy output image name, the name finally '{save_name}-seq.{output_format}', example: 'image_name-0.png'")
    outpaint_distance: List[int] = Field(default=[0, 0, 0, 0], description="Outpaint Distance, number in list means [left, top, right, bottom]")
    upscale_multiple: float = Field(default=1.0, ge=1.0, le=5.0, description="Upscale Rate, use only when uov_method is 'Upscale (Custom)'")
    preset: str = Field(default='initial', description="Presets")
    stream_output: bool = Field(default=False, description="Stream output")
    require_base64: bool = Field(default=False, description="Return base64 data of generated image")
    async_process: bool = Field(default=False, description="Set to true will run async and return job info for retrieve generation result later")
    webhook_url: str | None = Field(default='', description="Optional URL for a webhook callback. If provided, the system will send a POST request to this URL upon task completion or failure."
                                                            " This allows for asynchronous notification of task status.")


class DescribeImageRequest(BaseModel):
    image: str = Field(description="Image url or base64")
    image_type: DescribeImageType = Field(default=DescribeImageType.photo, description="Image type, 'Photo' or 'Anime'")
