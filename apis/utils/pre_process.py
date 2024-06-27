"""
Pre-process parameters
"""
import copy
import os
import random

from modules.config import default_max_lora_number
from modules.flags import controlnet_image_count
from modules import constants
from apis.models.requests import CommonRequest
from apis.utils.file_utils import save_base64
from apis.utils.img_utils import read_input_image
from apis.models.base import Lora, ImagePrompt


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, '..', 'inputs')
OUT_PATH = os.path.join(ROOT_DIR, '..', 'outputs')


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


async def pre_worker(request: CommonRequest):
    """
    Pre-processes the request.
    :param request: The request to pre-process.
    :return: The pre-processed request.
    """
    os.makedirs(INPUT_PATH, exist_ok=True)

    request.aspect_ratios_selection = aspect_ratios_parser(request.aspect_ratios_selection)
    request.image_seed = refresh_seed(request.image_seed)

    request.input_image_checkbox = True
    request.inpaint_mask_upload_checkbox = True
    request.invert_mask_checkbox = False

    request.uov_input_image = await read_input_image(request.uov_input_image)
    request.inpaint_input_image = await read_input_image(request.inpaint_input_image)
    request.inpaint_mask_image_upload = await read_input_image(request.inpaint_mask_image_upload)

    request.loras = lora_parser(request.loras)
    request.controlnet_image = control_net_parser(request.controlnet_image)

    if request.controlnet_image[0] is not None:
        request.current_tab = 'ip'
    elif request.uov_input_image is not None:
        request.current_tab = 'uov'
    elif request.inpaint_input_image is not None:
        request.current_tab = 'inpaint'

    req_copy = copy.deepcopy(request)

    request.inpaint_input_image = {
        "image": request.inpaint_input_image,
        "mask": request.inpaint_mask_image_upload
    }

    req_copy.uov_input_image = save_base64(req_copy.uov_input_image, INPUT_PATH)
    req_copy.inpaint_input_image = save_base64(req_copy.inpaint_input_image, INPUT_PATH)
    req_copy.inpaint_mask_image_upload = save_base64(req_copy.inpaint_mask_image_upload, INPUT_PATH)

    cn_imgs = []
    controlnet_images = [list(group) for group in zip(*[iter(req_copy.controlnet_image)]*4)]
    for cn in controlnet_images:
        control_net = {
            "cn_img": save_base64(cn[0], INPUT_PATH),
            "cn_stop": cn[1],
            "cn_weight": cn[2],
            "cn_type": cn[3]
        }
        cn_imgs.append(control_net)
    req_copy.controlnet_image = cn_imgs
    return req_copy, request
