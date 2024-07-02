"""
API utils
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from apis.models.requests import CommonRequest
from apis.models.base import UpscaleOrVaryMethod

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
APIKEY_AUTH = None


def api_key_auth(apikey: str = Security(api_key_header)):
    """
    Check if the API key is valid, API key is not required if no API key is set
    Args:
        apikey: API key
    returns:
        None if API key is not set, otherwise raise HTTPException
    """
    if APIKEY_AUTH is None:
        return  # Skip API key check if no API key is set
    if apikey != APIKEY_AUTH:
        raise HTTPException(status_code=403, detail="Forbidden")


def params_to_params(req: CommonRequest):
    """
    Convert params to params string
    """
    uov_method = req.uov_method.value
    if req.uov_method == UpscaleOrVaryMethod.upscale_custom:
        uov_method = f"Upscale ({req.upscale_rate}x)"
    params = [
        req.generate_image_grid,
        req.prompt,
        req.negative_prompt,
        req.style_selections,
        req.performance_selection.value,
        req.aspect_ratios_selection,
        req.image_number,
        req.output_format,
        req.image_seed,
        req.read_wildcards_in_order,
        req.sharpness,
        req.guidance_scale,
        req.base_model_name,
        req.refiner_model_name,
        req.refiner_switch
    ]
    params.extend(req.loras)
    params.extend([
        req.input_image_checkbox,
        req.current_tab,
        uov_method,
        req.uov_input_image,
        req.outpaint_selections,
        req.inpaint_input_image,
        req.inpaint_additional_prompt,
        req.inpaint_mask_image_upload,

        req.disable_preview,
        req.disable_intermediate_results,
        req.disable_seed_increment,
        req.black_out_nsfw,
        req.adm_scaler_positive,
        req.adm_scaler_negative,
        req.adm_scaler_end,
        req.adaptive_cfg,
        req.clip_skip,
        req.sampler_name,
        req.scheduler_name,
        req.vae_name,
        req.overwrite_step,
        req.overwrite_switch,
        req.overwrite_width,
        req.overwrite_height,
        req.overwrite_vary_strength,
        req.overwrite_upscale_strength,
        req.mixing_image_prompt_and_vary_upscale,
        req.mixing_image_prompt_and_inpaint,
        req.debugging_cn_preprocessor,
        req.skipping_cn_preprocessor,
        req.canny_low_threshold,
        req.canny_high_threshold,
        req.refiner_swap_method,
        req.controlnet_softness,
        req.freeu_enabled,
        req.freeu_b1,
        req.freeu_b2,
        req.freeu_s1,
        req.freeu_s2,
        req.debugging_inpaint_preprocessor,
        req.inpaint_disable_initial_latent,
        req.inpaint_engine,
        req.inpaint_strength,
        req.inpaint_respective_field,
        req.inpaint_mask_upload_checkbox,
        req.invert_mask_checkbox,
        req.inpaint_erode_or_dilate,

        req.save_metadata_to_images,
        req.metadata_scheme.value,
    ])
    params.extend(req.controlnet_image)
    return params
