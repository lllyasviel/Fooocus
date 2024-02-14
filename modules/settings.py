import json
import os
import math
import numbers
import re
from attrs import asdict, define, field, validators
from pathlib import Path

import args_manager
import modules.flags
import modules.sdxl_styles

from modules.model_loader import load_file_from_url
from modules.util import get_files_from_folder


def load_settings():
    config_path = Path("./config.txt").absolute()

    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(SettingsDefault()), file, indent=4)

    settings = SettingsDefault()
    settings.update_with_file(config_path)

    preset_path = Path(f'./presets/{args_manager.args.preset}.json').absolute()
    if preset_path.exists():
        settings.update_with_file(preset_path)

    return settings


def validator_path_exist(instance, attribute, value):
    if not Path(value).exists():
        raise ValueError(f"Path {value} does not exist")


def validator_list_of_strings(instance, attribute, value):
    if not (isinstance(value, list) and all(isinstance(elem, str) for elem in value)):
        raise ValueError(f"Value {value} is not list of strings")


def validator_legal_styles(instance, attribute, value):
    if not all(style in modules.sdxl_styles.legal_style_names for style in value):
        raise ValueError(f"Value {value} is not list of legal styles")


def validator_list_of_resolutions(instance, attribute, value):
    resolution_pattern = re.compile(r'^\d+\*\d+$')
    for resolution in value:
        if not resolution_pattern.match(resolution):
            raise ValueError(f"Value {resolution} is not a valid resolution")


_default_resolutions = [
    '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'
]


@define
class SettingsDefault:
    # paths
    path_checkpoints = field(default="./models/checkpoints/", validator=validator_path_exist)
    path_loras = field(default="./models/loras/", validator=validator_path_exist)
    path_embeddings = field(default="./models/embeddings/", validator=validator_path_exist)
    path_vae_approx = field(default="./models/vae_approx/", validator=validator_path_exist)
    path_upscale_models = field(default="./models/upscale_models/", validator=validator_path_exist)
    path_inpaint = field(default="./models/inpaint/", validator=validator_path_exist)
    path_controlnet = field(default="./models/controlnet/", validator=validator_path_exist)
    path_clip_vision = field(default="./models/clip_vision/", validator=validator_path_exist)
    path_fooocus_expansion = field(default="./models/prompt_expansion/", validator=validator_path_exist)
    path_outputs = field(default="./outputs/")
    # models
    default_base_model_name = field(default="juggernautXL_v8Rundiffusion.safetensors",
                                    validator=validators.instance_of(str))
    previous_default_models = field(default=[], validator=validators.instance_of(list))
    default_refiner_model_name = field(default="None", validator=validators.instance_of(str))
    default_refiner_switch = field(default=0.5, validator=[validators.ge(0), validators.le(1)])
    # loras
    lora1_name = field(default="None", validator=validators.instance_of(str))
    lora1_weight = field(default=1.0, validator=[validators.ge(-2), validators.le(2)])
    lora2_name = field(default="None", validator=validators.instance_of(str))
    lora2_weight = field(default=1.0, validator=[validators.ge(-2), validators.le(2)])
    lora3_name = field(default="None", validator=validators.instance_of(str))
    lora3_weight = field(default=1.0, validator=[validators.ge(-2), validators.le(2)])
    lora4_name = field(default="None", validator=validators.instance_of(str))
    lora4_weight = field(default=1.0, validator=[validators.ge(-2), validators.le(2)])
    lora5_name = field(default="None", validator=validators.instance_of(str))
    lora5_weight = field(default=1.0, validator=[validators.ge(-2), validators.le(2)])
    # config
    default_cfg_scale = field(default=7.0, validator=validators.instance_of(numbers.Number))
    default_sample_sharpness = field(default=2.0, validator=validators.instance_of(numbers.Number))
    default_sampler = field(default="dpmpp_2m_sde_gpu", validator=validators.in_(modules.flags.sampler_list))
    default_scheduler = field(default="karras", validator=validators.in_(modules.flags.scheduler_list))
    default_styles = field(default=["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"], validator=validator_legal_styles)
    default_prompt = field(default="", validator=validators.instance_of(str))
    default_prompt_negative = field(default="", validator=validators.instance_of(str))
    default_performance = field(default="Speed", validator=validators.in_(modules.flags.performance_selections))
    default_advanced_checkbox = field(default=False, validator=validators.instance_of(bool))
    default_max_image_number = field(default=32, validator=[validators.instance_of(int), validators.ge(1)])
    default_image_number = field(default=2, validator=[validators.instance_of(int), validators.ge(1)])
    checkpoint_downloads = field(default={}, validator=validators.instance_of(dict))
    lora_downloads = field(default={}, validator=validators.instance_of(dict))
    embeddings_downloads = field(default={}, validator=validators.instance_of(dict))
    available_aspect_ratios = field(default=_default_resolutions, validator=validator_list_of_resolutions)
    default_aspect_ratio = field(default='1152*896', validator=validators.matches_re(r'^\d+\*\d+$'))
    default_inpaint_engine_version = field(default="v2.6", validator=validators.in_(modules.flags.inpaint_engine_versions))
    default_cfg_tsnr = field(default=7.0, validator=validators.instance_of(numbers.Number))
    default_overwrite_step = field(default=-1, validator=validators.instance_of(int))
    default_overwrite_switch = field(default=-1, validator=validators.instance_of(int))

    def update_key(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            print(f"Invalid attribute name: {key}. Ignoring")

    def update_with_file(self, path):
        cfg = {}
        with path.open("r", encoding="utf-8") as file:
            cfg.update(json.load(file))

        for k, v in cfg.items():
            try:
                self.update_key(k, v)
            except ValueError as e:
                print(e)
                print(f"Error: key {k} has invalid value {v}. Will use default {getattr(self, k)}")


def get_model_filenames(folder_path, name_filter=None):
    return get_files_from_folder(folder_path, ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'], name_filter)


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(settings.path_checkpoints)
    lora_filenames = get_model_filenames(settings.path_loras)
    return


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=settings.path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(settings.path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=settings.path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(settings.path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=settings.path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(settings.path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=settings.path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(settings.path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=settings.path_loras,
        file_name='sdxl_lcm_lora.safetensors'
    )
    return 'sdxl_lcm_lora.safetensors'


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=settings.path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(settings.path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=settings.path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(settings.path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=settings.path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(settings.path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=settings.path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(settings.path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=settings.path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(settings.path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=settings.path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(settings.path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=settings.path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(settings.path_upscale_models, 'fooocus_upscaler_s409985e5.bin')


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {a // g}:{b // g}</span>'


settings = load_settings()
os.makedirs(settings.path_outputs, exist_ok=True)

model_filenames = []
lora_filenames = []
update_all_model_names()

default_aspect_ratio = add_ratio(settings.default_aspect_ratio)
available_aspect_ratios = [add_ratio(x) for x in settings.available_aspect_ratios]
