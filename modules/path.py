import os
import json
import re
from modules.model_loader import load_file_from_url


config_path = "user_path_config.txt"
config_dict = {}
filter = re.compile(r'[Xx][Ll]')

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict = json.load(json_file)
except Exception as e:
    print('Load path config failed')
    print(e)


def get_config_or_set_default(key, default):
    global config_dict
    v = config_dict.get(key, None)
    if isinstance(v, str) and os.path.exists(v) and os.path.isdir(v):
        return v
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default))
        os.makedirs(dp, exist_ok=True)
        config_dict[key] = dp
        return dp


modelfile_path = get_config_or_set_default('modelfile_path', '../models/checkpoints/')
lorafile_path = get_config_or_set_default('lorafile_path', '../models/loras/')
vae_approx_path = get_config_or_set_default('vae_approx_path', '../models/vae_approx/')
upscale_models_path = get_config_or_set_default('upscale_models_path', '../models/upscale_models/')
inpaint_models_path = get_config_or_set_default('inpaint_models_path', '../models/inpaint/')
fooocus_expansion_path = get_config_or_set_default('fooocus_expansion_path',
                                                   '../models/prompt_expansion/fooocus_expansion')

temp_outputs_path = get_config_or_set_default('temp_outputs_path', '../outputs/')

with open(config_path, "w", encoding="utf-8") as json_file:
    json.dump(config_dict, json_file, indent=4)


os.makedirs(temp_outputs_path, exist_ok=True)

model_filenames = []
lora_filenames = []


def get_model_filenames(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            for ends in ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']:
                if filename.lower().endswith(ends) and filter.search(filename):
                    filenames.append(filename)
                    break
    return filenames


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    return

update_all_model_names()

'''def sdxl_model_fitler(model_filenames,lora_filenames):
    print(model_filenames)
    for model in model_filenames:
        if filter.match(model) == None:
            print(model)
            model_filenames.remove(model)
    print(model_filenames)
    print(lora_filenames)
    for lora in lora_filenames:
        if filter.match(lora) == None:
            print(lora)
            model_filenames.remove(lora)
    print(lora_filenames)
'''

default_base_model_name = model_filenames[0]
default_refiner_model_name = model_filenames[0]
default_lora_name = lora_filenames[0]
default_lora_weight = 0.5

def downloading_inpaint_models():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=inpaint_models_path,
        file_name='fooocus_inpaint_head.pth'
    )
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
        model_dir=inpaint_models_path,
        file_name='inpaint.fooocus.patch'
    )
    return os.path.join(inpaint_models_path, 'fooocus_inpaint_head.pth'), os.path.join(inpaint_models_path, 'inpaint.fooocus.patch')


update_all_model_names()
