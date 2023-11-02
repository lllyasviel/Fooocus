import os
import json
import args_manager
import modules.flags
import modules.sdxl_styles

from modules.model_loader import load_file_from_url
from modules.util import get_files_from_folder

config_path = "user_path_config.txt"
config_dict = {}
visited_keys = []

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict = json.load(json_file)
except Exception as e:
    print('Load path config failed')
    print(e)

preset = args_manager.args.preset

if isinstance(preset, str):
    preset = os.path.abspath(f'./presets/{preset}.json')
    try:
        if os.path.exists(preset):
            with open(preset, "r", encoding="utf-8") as json_file:
                preset = json.load(json_file)
    except Exception as e:
        print('Load preset config failed')
        print(e)

preset = preset if isinstance(preset, dict) else None

if preset is not None:
    config_dict.update(preset)


def get_dir_or_set_default(key, default_value):
    global config_dict, visited_keys
    visited_keys.append(key)
    v = config_dict.get(key, None)
    if isinstance(v, str) and os.path.exists(v) and os.path.isdir(v):
        return v
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        config_dict[key] = dp
        return dp


modelfile_path = get_dir_or_set_default('modelfile_path', '../models/checkpoints/')
lorafile_path = get_dir_or_set_default('lorafile_path', '../models/loras/')
embeddings_path = get_dir_or_set_default('embeddings_path', '../models/embeddings/')
vae_approx_path = get_dir_or_set_default('vae_approx_path', '../models/vae_approx/')
upscale_models_path = get_dir_or_set_default('upscale_models_path', '../models/upscale_models/')
inpaint_models_path = get_dir_or_set_default('inpaint_models_path', '../models/inpaint/')
controlnet_models_path = get_dir_or_set_default('controlnet_models_path', '../models/controlnet/')
clip_vision_models_path = get_dir_or_set_default('clip_vision_models_path', '../models/clip_vision/')
fooocus_expansion_path = get_dir_or_set_default('fooocus_expansion_path',
                                                '../models/prompt_expansion/fooocus_expansion')
temp_outputs_path = get_dir_or_set_default('temp_outputs_path', '../outputs/')


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False):
    global config_dict, visited_keys
    visited_keys.append(key)
    if key not in config_dict:
        config_dict[key] = default_value
        return default_value

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        config_dict[key] = default_value
        return default_value


default_base_model_name = get_config_item_or_set_default(
    key='default_model',
    default_value='juggernautXL_version6Rundiffusion.safetensors',
    validator=lambda x: isinstance(x, str)
)
default_refiner_model_name = get_config_item_or_set_default(
    key='default_refiner',
    default_value='None',
    validator=lambda x: isinstance(x, str)
)
default_refiner_switch = get_config_item_or_set_default(
    key='default_refiner_switch',
    default_value=0.8,
    validator=lambda x: isinstance(x, float)
)
default_lora_name = get_config_item_or_set_default(
    key='default_lora',
    default_value='sd_xl_offset_example-lora_1.0.safetensors',
    validator=lambda x: isinstance(x, str)
)
default_lora_weight = get_config_item_or_set_default(
    key='default_lora_weight',
    default_value=0.1,
    validator=lambda x: isinstance(x, float)
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=4.0,
    validator=lambda x: isinstance(x, float)
)
default_sample_sharpness = get_config_item_or_set_default(
    key='default_sample_sharpness',
    default_value=2,
    validator=lambda x: isinstance(x, float)
)
default_sampler = get_config_item_or_set_default(
    key='default_sampler',
    default_value='dpmpp_2m_sde_gpu',
    validator=lambda x: x in modules.flags.sampler_list
)
default_scheduler = get_config_item_or_set_default(
    key='default_scheduler',
    default_value='karras',
    validator=lambda x: x in modules.flags.scheduler_list
)
default_styles = get_config_item_or_set_default(
    key='default_styles',
    default_value=['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp'],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x)
)
default_prompt_negative = get_config_item_or_set_default(
    key='default_prompt_negative',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True
)
default_prompt = get_config_item_or_set_default(
    key='default_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True
)
default_advanced_checkbox = get_config_item_or_set_default(
    key='default_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool)
)
default_image_number = get_config_item_or_set_default(
    key='default_image_number',
    default_value=2,
    validator=lambda x: isinstance(x, int) and x >= 1 and x <= 32
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={
        'juggernautXL_version6Rundiffusion.safetensors':
            'https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors'
    },
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={
        'sd_xl_offset_example-lora_1.0.safetensors':
            'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors'
    },
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
available_aspect_ratios = get_config_item_or_set_default(
    key='available_aspect_ratios',
    default_value=['704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152', '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960', '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768', '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640', '1664*576', '1728*576'],
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value='1152*896' if '1152*896' in available_aspect_ratios else available_aspect_ratios[0],
    validator=lambda x: x in available_aspect_ratios
)

if preset is None:
    # Do not overwrite user config if preset is applied.
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

os.makedirs(temp_outputs_path, exist_ok=True)

model_filenames = []
lora_filenames = []

available_aspect_ratios = [x.replace('*', '×') for x in available_aspect_ratios]
default_aspect_ratio = default_aspect_ratio.replace('*', '×')


def get_model_filenames(folder_path, name_filter=None):
    return get_files_from_folder(folder_path, ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'], name_filter)


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    return


def downloading_inpaint_models(v):
    assert v in ['v1', 'v2.5']

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=inpaint_models_path,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(inpaint_models_path, 'fooocus_inpaint_head.pth')
    patch_file = None

    # load_file_from_url(
    #     url='https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth',
    #     model_dir=inpaint_models_path,
    #     file_name='ControlNetLama.pth'
    # )
    # lama_file = os.path.join(inpaint_models_path, 'ControlNetLama.pth')

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=inpaint_models_path,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(inpaint_models_path, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=inpaint_models_path,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(inpaint_models_path, 'inpaint_v25.fooocus.patch')

    return head_file, patch_file


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=controlnet_models_path,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(controlnet_models_path, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=controlnet_models_path,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(controlnet_models_path, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters():
    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=clip_vision_models_path,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(clip_vision_models_path, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=controlnet_models_path,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(controlnet_models_path, 'fooocus_ip_negative.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
        model_dir=controlnet_models_path,
        file_name='ip-adapter-plus_sdxl_vit-h.bin'
    )
    results += [os.path.join(controlnet_models_path, 'ip-adapter-plus_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=upscale_models_path,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(upscale_models_path, 'fooocus_upscaler_s409985e5.bin')


update_all_model_names()
