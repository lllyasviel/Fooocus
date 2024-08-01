"""
Pre-process parameters
"""
import copy
import os
import random

import numpy as np

from modules.config import (
    default_max_lora_number,
    try_get_preset_content,
    default_controlnet_image_count
)
from modules.flags import Performance
from modules import constants, config
from modules.model_loader import load_file_from_url

from apis.models.requests import CommonRequest
from apis.utils.file_utils import save_base64, to_http
from apis.utils.img_utils import read_input_image
from apis.models.base import EnhanceCtrlNets, Lora, ImagePrompt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, '..', 'inputs')


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


async def control_net_parser(control_net: list) -> list:
    """
    Convert control net to control net list
    """
    default_cn_image = {
        "cn_img": None,
        "cn_stop": 0.6,
        "cn_weight": 0.6,
        "cn_type": "ImagePrompt"
    }
    while len(control_net) < default_controlnet_image_count:
        control_net.append(ImagePrompt(**default_cn_image))

    control_net = control_net[:default_controlnet_image_count]
    cn_list = []
    for cn in control_net:
        cn_list.extend([
            await read_input_image(cn.cn_img),
            cn.cn_stop,
            cn.cn_weight,
            cn.cn_type.value
        ])
    return cn_list


def parse_preset(request: CommonRequest) -> CommonRequest:
    """
    Parse preset content.
    :param request: The request to parse.
    :return: The parsed request.
    """
    default_params = CommonRequest()
    preset_content = try_get_preset_content(request.preset)

    if preset_content != {}:
        request.prompt = preset_content.get("default_prompt") if request.prompt == default_params.prompt else request.prompt
        request.negative_prompt = preset_content.get("default_prompt_negative") if request.negative_prompt == default_params.negative_prompt else request.negative_prompt
        request.base_model_name = preset_content.get("default_model") if request.base_model_name == default_params.base_model_name else request.base_model_name
        request.refiner_model_name = preset_content.get("default_refiner") if request.refiner_model_name == default_params.refiner_model_name else request.refiner_model_name
        request.refiner_switch = preset_content.get("default_refiner_switch") if request.refiner_switch == default_params.refiner_switch else request.refiner_switch
        request.style_selections = preset_content.get("default_styles") if request.style_selections == default_params.style_selections else request.style_selections
        request.adaptive_cfg = preset_content.get("default_cfg_scale") if request.adaptive_cfg == default_params.adaptive_cfg else request.adaptive_cfg
        request.sharpness = preset_content.get("default_sample_sharpness") if request.sharpness == default_params.sharpness else request.sharpness
        request.sampler_name = preset_content.get("default_sampler") if request.sampler_name == default_params.sampler_name else request.sampler_name
        request.scheduler_name = preset_content.get("default_scheduler") if request.scheduler_name == default_params.scheduler_name else request.scheduler_name
        request.performance_selection = Performance(preset_content.get("default_performance")) if request.performance_selection == default_params.performance_selection else request.performance_selection
        request.aspect_ratios_selection = preset_content.get("default_aspect_ratio") if request.aspect_ratios_selection == default_params.aspect_ratios_selection else request.aspect_ratios_selection
        request.vae_name = preset_content.get("default_vae", "Default (model)") if request.vae_name == default_params.vae_name else request.vae_name

        checkpoint_downloads = preset_content.get("checkpoint_downloads", {})
        embeddings_downloads = preset_content.get("embeddings_downloads", {})
        lora_downloads = preset_content.get("lora_downloads", {})
        vae_downloads = preset_content.get("vae_downloads", {})
        for file_name, url in checkpoint_downloads.items():
            load_file_from_url(url=url, model_dir=config.paths_checkpoints[0], file_name=file_name)
        for file_name, url in embeddings_downloads.items():
            load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
        for file_name, url in lora_downloads.items():
            load_file_from_url(url=url, model_dir=config.paths_loras[0], file_name=file_name)
        for file_name, url in vae_downloads.items():
            load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)
    return request


async def pre_worker(request: CommonRequest):
    """
    Pre-processes the request.
    :param request: The request to pre-process.
    :return: The pre-processed request.
    """
    if request.preset != 'initial':
        request = parse_preset(request)

    os.makedirs(INPUT_PATH, exist_ok=True)

    request.aspect_ratios_selection = aspect_ratios_parser(request.aspect_ratios_selection)
    request.image_seed = refresh_seed(request.image_seed)

    request.input_image_checkbox = True
    request.inpaint_advanced_masking_checkbox = True
    if request.inpaint_mask_image_upload is None or request.inpaint_mask_image_upload == 'None':
        request.inpaint_advanced_masking_checkbox = False
    request.invert_mask_checkbox = False

    request.uov_input_image = await read_input_image(request.uov_input_image)
    request.inpaint_input_image = await read_input_image(request.inpaint_input_image)
    request.inpaint_mask_image_upload = await read_input_image(request.inpaint_mask_image_upload)

    request.loras = lora_parser(request.loras)
    request.controlnet_image = await control_net_parser(request.controlnet_image)
    request.enhance_input_image = await read_input_image(request.enhance_input_image)

    if request.enhance_input_image is not None:
        request.current_tab = 'enhance'
    elif request.controlnet_image[0] is not None:
        request.current_tab = 'ip'
    elif request.uov_input_image is not None:
        request.current_tab = 'uov'
    elif request.inpaint_input_image is not None:
        request.current_tab = 'inpaint'

    while len(request.enhance_ctrls) < 3:
        request.enhance_ctrls.append(EnhanceCtrlNets())

    req_copy = copy.deepcopy(request)
    if request.inpaint_mask_image_upload is None and request.inpaint_input_image is not None:
        inpaint_image_size = request.inpaint_input_image.shape[:3]
        request.inpaint_mask_image_upload = np.zeros(inpaint_image_size, dtype=np.uint8)
    request.inpaint_input_image = {
        "image": request.inpaint_input_image,
        "mask": request.inpaint_mask_image_upload
    }

    req_copy.uov_input_image = to_http(save_base64(req_copy.uov_input_image, INPUT_PATH), "inputs")
    req_copy.inpaint_input_image = to_http(save_base64(req_copy.inpaint_input_image, INPUT_PATH), "inputs")
    req_copy.inpaint_mask_image_upload = to_http(save_base64(req_copy.inpaint_mask_image_upload, INPUT_PATH), "inputs")
    req_copy.enhance_input_image = to_http(save_base64(req_copy.enhance_input_image, INPUT_PATH), "inputs")

    cn_imgs = []
    controlnet_images = [list(group) for group in zip(*[iter(req_copy.controlnet_image)]*4)]
    for cn in controlnet_images:
        control_net = {
            "cn_img": to_http(save_base64(cn[0], INPUT_PATH), "inputs"),
            "cn_stop": cn[1],
            "cn_weight": cn[2],
            "cn_type": cn[3]
        }
        cn_imgs.append(control_net)
    req_copy.controlnet_image = cn_imgs
    return req_copy, request
