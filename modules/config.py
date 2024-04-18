import os
import json
import math
import numbers
import args_manager
import tempfile
import modules.flags
import modules.sdxl_styles

from modules.model_loader import load_file_from_url
from modules.util import get_files_from_folder, makedirs_with_log
from modules.flags import OutputFormat, Performance, MetadataScheme


def get_config_path(key, default_value):
    env = os.getenv(key)
    if env is not None and isinstance(env, str):
        print(f"Environment: {key} = {env}")
        return env
    else:
        return os.path.abspath(default_value)


config_path = get_config_path('config_path', "./config.txt")
config_example_path = get_config_path('config_example_path', "config_modification_tutorial.txt")
config_dict = {}
always_save_keys = []
visited_keys = []

try:
    with open(os.path.abspath(f'./presets/default.json'), "r", encoding="utf-8") as json_file:
        config_dict.update(json.load(json_file))
except Exception as e:
    print(f'Load default preset failed.')
    print(e)

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict.update(json.load(json_file))
            always_save_keys = list(config_dict.keys())
except Exception as e:
    print(f'Failed to load config file "{config_path}" . The reason is: {str(e)}')
    print('Please make sure that:')
    print(f'1. The file "{config_path}" is a valid text file, and you have access to read it.')
    print('2. Use "\\\\" instead of "\\" when describing paths.')
    print('3. There is no "," before the last "}".')
    print('4. All key/value formats are correct.')


def try_load_deprecated_user_path_config():
    global config_dict

    if not os.path.exists('user_path_config.txt'):
        return

    try:
        deprecated_config_dict = json.load(open('user_path_config.txt', "r", encoding="utf-8"))

        def replace_config(old_key, new_key):
            if old_key in deprecated_config_dict:
                config_dict[new_key] = deprecated_config_dict[old_key]
                del deprecated_config_dict[old_key]

        replace_config('modelfile_path', 'path_checkpoints')
        replace_config('lorafile_path', 'path_loras')
        replace_config('embeddings_path', 'path_embeddings')
        replace_config('vae_approx_path', 'path_vae_approx')
        replace_config('upscale_models_path', 'path_upscale_models')
        replace_config('inpaint_models_path', 'path_inpaint')
        replace_config('controlnet_models_path', 'path_controlnet')
        replace_config('clip_vision_models_path', 'path_clip_vision')
        replace_config('fooocus_expansion_path', 'path_fooocus_expansion')
        replace_config('temp_outputs_path', 'path_outputs')

        if deprecated_config_dict.get("default_model", None) == 'juggernautXL_version6Rundiffusion.safetensors':
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully in silence. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return

        if input("Newer models and configs are available. "
                 "Download and update files? [Y/n]:") in ['n', 'N', 'No', 'no', 'NO']:
            config_dict.update(deprecated_config_dict)
            print('Loading using deprecated old models and deprecated old configs.')
            return
        else:
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully by user. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return
    except Exception as e:
        print('Processing deprecated config failed')
        print(e)
    return


try_load_deprecated_user_path_config()


def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index('.json')] for f in os.listdir(preset_folder) if f.endswith('.json')]


def try_get_preset_content(preset):
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    json_content = json.load(json_file)
                    print(f'Loaded preset: {preset_path}')
                    return json_content
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    return {}

available_presets = get_presets()
preset = args_manager.args.preset
config_dict.update(try_get_preset_content(preset))

def get_path_output() -> str:
    """
    Checking output path argument and overriding default path.
    """
    global config_dict
    path_output = get_dir_or_set_default('path_outputs', '../outputs/', make_directory=True)
    if args_manager.args.output_path:
        print(f'Overriding config value path_outputs with {args_manager.args.output_path}')
        config_dict['path_outputs'] = path_output = args_manager.args.output_path
    return path_output


def get_dir_or_set_default(key, default_value, as_array=False, make_directory=False):
    global config_dict, visited_keys, always_save_keys

    if key not in visited_keys:
        visited_keys.append(key)

    if key not in always_save_keys:
        always_save_keys.append(key)

    v = os.getenv(key)
    if v is not None:
        print(f"Environment: {key} = {v}")
        config_dict[key] = v
    else:
        v = config_dict.get(key, None)

    if isinstance(v, str):
        if make_directory:
            makedirs_with_log(v)
        if os.path.exists(v) and os.path.isdir(v):
            return v if not as_array else [v]
    elif isinstance(v, list):
        if make_directory:
            for d in v:
                makedirs_with_log(d)
        if all([os.path.exists(d) and os.path.isdir(d) for d in v]):
            return v

    if v is not None:
        print(f'Failed to load config key: {json.dumps({key:v})} is invalid or does not exist; will use {json.dumps({key:default_value})} instead.')
    if isinstance(default_value, list):
        dp = []
        for path in default_value:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            dp.append(abs_path)
            os.makedirs(abs_path, exist_ok=True)
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        if as_array:
            dp = [dp]
    config_dict[key] = dp
    return dp


paths_checkpoints = get_dir_or_set_default('path_checkpoints', ['../models/checkpoints/'], True)
paths_loras = get_dir_or_set_default('path_loras', ['../models/loras/'], True)
path_embeddings = get_dir_or_set_default('path_embeddings', '../models/embeddings/')
path_vae_approx = get_dir_or_set_default('path_vae_approx', '../models/vae_approx/')
path_upscale_models = get_dir_or_set_default('path_upscale_models', '../models/upscale_models/')
path_inpaint = get_dir_or_set_default('path_inpaint', '../models/inpaint/')
path_controlnet = get_dir_or_set_default('path_controlnet', '../models/controlnet/')
path_clip_vision = get_dir_or_set_default('path_clip_vision', '../models/clip_vision/')
path_fooocus_expansion = get_dir_or_set_default('path_fooocus_expansion', '../models/prompt_expansion/fooocus_expansion')
path_wildcards = get_dir_or_set_default('path_wildcards', '../wildcards/')
path_outputs = get_path_output()


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False):
    global config_dict, visited_keys

    if key not in visited_keys:
        visited_keys.append(key)
    
    v = os.getenv(key)
    if v is not None:
        print(f"Environment: {key} = {v}")
        config_dict[key] = v

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
        if v is not None:
            print(f'Failed to load config key: {json.dumps({key:v})} is invalid; will use {json.dumps({key:default_value})} instead.')
        config_dict[key] = default_value
        return default_value


def init_temp_path(path: str | None, default_path: str) -> str:
    if args_manager.args.temp_path:
        path = args_manager.args.temp_path

    if path != '' and path != default_path:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            print(f'Using temp path {path}')
            return path
        except Exception as e:
            print(f'Could not create temp path {path}. Reason: {e}')
            print(f'Using default temp path {default_path} instead.')

    os.makedirs(default_path, exist_ok=True)
    return default_path


default_temp_path = os.path.join(tempfile.gettempdir(), 'fooocus')
temp_path = init_temp_path(get_config_item_or_set_default(
    key='temp_path',
    default_value=default_temp_path,
    validator=lambda x: isinstance(x, str),
), default_temp_path)
temp_path_cleanup_on_launch = get_config_item_or_set_default(
    key='temp_path_cleanup_on_launch',
    default_value=True,
    validator=lambda x: isinstance(x, bool)
)
default_base_model_name = default_model = get_config_item_or_set_default(
    key='default_model',
    default_value='model.safetensors',
    validator=lambda x: isinstance(x, str)
)
previous_default_models = get_config_item_or_set_default(
    key='previous_default_models',
    default_value=[],
    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x)
)
default_refiner_model_name = default_refiner = get_config_item_or_set_default(
    key='default_refiner',
    default_value='None',
    validator=lambda x: isinstance(x, str)
)
default_refiner_switch = get_config_item_or_set_default(
    key='default_refiner_switch',
    default_value=0.8,
    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1
)
default_loras_min_weight = get_config_item_or_set_default(
    key='default_loras_min_weight',
    default_value=-2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10
)
default_loras_max_weight = get_config_item_or_set_default(
    key='default_loras_max_weight',
    default_value=2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10
)
default_loras = get_config_item_or_set_default(
    key='default_loras',
    default_value=[
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ]
    ],
    validator=lambda x: isinstance(x, list) and all(
        len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
        or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
        for y in x)
)
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = get_config_item_or_set_default(
    key='default_max_lora_number',
    default_value=len(default_loras) if isinstance(default_loras, list) and len(default_loras) > 0 else 5,
    validator=lambda x: isinstance(x, int) and x >= 1
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number)
)
default_sample_sharpness = get_config_item_or_set_default(
    key='default_sample_sharpness',
    default_value=2.0,
    validator=lambda x: isinstance(x, numbers.Number)
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
    default_value=[
        "Fooocus V2",
        "Fooocus Enhance",
        "Fooocus Sharp"
    ],
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
default_performance = get_config_item_or_set_default(
    key='default_performance',
    default_value=Performance.SPEED.value,
    validator=lambda x: x in Performance.list()
)
default_advanced_checkbox = get_config_item_or_set_default(
    key='default_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool)
)
default_max_image_number = get_config_item_or_set_default(
    key='default_max_image_number',
    default_value=32,
    validator=lambda x: isinstance(x, int) and x >= 1
)
default_output_format = get_config_item_or_set_default(
    key='default_output_format',
    default_value='png',
    validator=lambda x: x in OutputFormat.list()
)
default_image_number = get_config_item_or_set_default(
    key='default_image_number',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= default_max_image_number
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
available_aspect_ratios = get_config_item_or_set_default(
    key='available_aspect_ratios',
    default_value=[
        '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
        '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
        '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
        '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
        '1664*576', '1728*576'
    ],
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value='1152*896' if '1152*896' in available_aspect_ratios else available_aspect_ratios[0],
    validator=lambda x: x in available_aspect_ratios
)
default_inpaint_engine_version = get_config_item_or_set_default(
    key='default_inpaint_engine_version',
    default_value='v2.6',
    validator=lambda x: x in modules.flags.inpaint_engine_versions
)
default_cfg_tsnr = get_config_item_or_set_default(
    key='default_cfg_tsnr',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number)
)
default_overwrite_step = get_config_item_or_set_default(
    key='default_overwrite_step',
    default_value=-1,
    validator=lambda x: isinstance(x, int)
)
default_overwrite_switch = get_config_item_or_set_default(
    key='default_overwrite_switch',
    default_value=-1,
    validator=lambda x: isinstance(x, int)
)
example_inpaint_prompts = get_config_item_or_set_default(
    key='example_inpaint_prompts',
    default_value=[
        'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
    ],
    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x)
)
default_save_metadata_to_images = get_config_item_or_set_default(
    key='default_save_metadata_to_images',
    default_value=False,
    validator=lambda x: isinstance(x, bool)
)
default_metadata_scheme = get_config_item_or_set_default(
    key='default_metadata_scheme',
    default_value=MetadataScheme.FOOOCUS.value,
    validator=lambda x: x in [y[1] for y in modules.flags.metadata_scheme if y[1] == x]
)
metadata_created_by = get_config_item_or_set_default(
    key='metadata_created_by',
    default_value='',
    validator=lambda x: isinstance(x, str)
)

example_inpaint_prompts = [[x] for x in example_inpaint_prompts]

config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [[True, 'None', 1.0] for _ in range(default_max_lora_number - len(default_loras))]

# mapping config to meta parameter 
possible_preset_keys = {
    "default_model": "base_model",
    "default_refiner": "refiner_model",
    "default_refiner_switch": "refiner_switch",
    "previous_default_models": "previous_default_models",
    "default_loras_min_weight": "default_loras_min_weight",
    "default_loras_max_weight": "default_loras_max_weight",
    "default_loras": "<processed>",
    "default_cfg_scale": "guidance_scale",
    "default_sample_sharpness": "sharpness",
    "default_sampler": "sampler",
    "default_scheduler": "scheduler",
    "default_overwrite_step": "steps",
    "default_performance": "performance",
    "default_image_number": "image_number",
    "default_prompt": "prompt",
    "default_prompt_negative": "negative_prompt",
    "default_styles": "styles",
    "default_aspect_ratio": "resolution",
    "default_save_metadata_to_images": "default_save_metadata_to_images",
    "checkpoint_downloads": "checkpoint_downloads",
    "embeddings_downloads": "embeddings_downloads",
    "lora_downloads": "lora_downloads"
}

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(args_manager.args.preset, str):
    save_path = 'presets/' + args_manager.args.preset + '.json'
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in possible_preset_keys}, json_file, indent=4)
    print(f'Preset saved to {save_path}. Exiting ...')
    exit(0)


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {a // g}:{b // g}</span>'


default_aspect_ratio = add_ratio(default_aspect_ratio)
available_aspect_ratios = [add_ratio(x) for x in available_aspect_ratios]


# Only write config in the first launch.
if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in always_save_keys}, json_file, indent=4)


# Always write tutorials.
with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

model_filenames = []
lora_filenames = []
wildcard_filenames = []

sdxl_lcm_lora = 'sdxl_lcm_lora.safetensors'
sdxl_lightning_lora = 'sdxl_lightning_4step_lora.safetensors'
loras_metadata_remove = [sdxl_lcm_lora, sdxl_lightning_lora]


def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
    files = []
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)
    return files


def update_files():
    global model_filenames, lora_filenames, wildcard_filenames, available_presets
    model_filenames = get_model_filenames(paths_checkpoints)
    lora_filenames = get_model_filenames(paths_loras)
    wildcard_filenames = get_files_from_folder(path_wildcards, ['.txt'])
    available_presets = get_presets()
    return


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=sdxl_lcm_lora
    )
    return sdxl_lcm_lora

def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=sdxl_lightning_lora
    )
    return sdxl_lightning_lora


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')


update_files()
