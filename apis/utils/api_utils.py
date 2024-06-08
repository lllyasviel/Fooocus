"""
API utils
"""
import random
from modules.config import default_max_lora_number
from modules.flags import controlnet_image_count
from modules import constants
from apis.models.requests import CommonRequest
from apis.models.base import Lora, ImagePrompt


def refresh_seed(seed_string: int | str | None) -> int:
    """
    Refresh and check seed number.
    :params seed_string: seed, str or int. None means random
    :return: seed number
    """
    if seed_string is None or seed_string == -1:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

    try:
        seed_value = int(seed_string)
        if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
            return seed_value
    except ValueError:
        pass
    return random.randint(constants.MIN_SEED, constants.MAX_SEED)


def aspect_ratios_parser(aspect_ratios: str) -> str:
    """
    Convert aspect ratios to aspect ratios string
    """
    return aspect_ratios.replace("*", "×")


def lora_parser(loras: list) -> list:
    """
    Convert loras to loras list
    """
    default_lora = {
        "enabled": True,
        "model_name": "None",
        "weight": 1,
    }
    while len(loras) < default_max_lora_number:
        loras.append(Lora(**default_lora))
    loras = loras[:default_max_lora_number]
    loras_list = []
    for lora in loras:
        loras_list.extend([
            lora.enabled,
            lora.model_name,
            lora.weight
        ])
    return loras_list


def control_net_parser(control_net: list) -> list:
    """
    Convert control net to control net list
    """
    default_cn_image = {
        "cn_img": None,
        "cn_stop": 0.6,
        "cn_weight": 0.6,
        "cn_type": "ImagePrompt"
    }
    while len(control_net) < controlnet_image_count:
        control_net.append(ImagePrompt(**default_cn_image))

    control_net = control_net[:controlnet_image_count]
    cn_list = []
    for cn in control_net:
        cn_list.extend([
            cn.cn_img,
            cn.cn_stop,
            cn.cn_weight,
            cn.cn_type.value
        ])
    return cn_list


def params_to_params(req: CommonRequest):
    """
    Convert params to params string
    """
    params = [
        req.generate_image_grid,
        req.prompt,
        req.negative_prompt,
        req.style_selections,
        req.performance_selection.value,
        aspect_ratios_parser(req.aspect_ratios_selection),
        req.image_number,
        req.output_format,
        refresh_seed(req.image_seed),
        req.read_wildcards_in_order,
        req.sharpness,
        req.guidance_scale,
        req.base_model_name,
        req.refiner_model_name,
        req.refiner_switch
    ]
    params.extend(lora_parser(req.loras))
    params.extend([
        req.input_image_checkbox,
        req.current_tab,
        req.uov_method.value,
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
    params.extend(control_net_parser(req.controlnet_image))
    return params
