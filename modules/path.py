import os
from modules.model_loader import load_file_from_url


modelfile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/checkpoints/'))
lorafile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/loras/'))
vae_approx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/vae_approx/'))
upscale_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/upscale_models/'))
inpaint_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/inpaint/'))
temp_outputs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/'))

fooocus_expansion_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                      '../models/prompt_expansion/fooocus_expansion'))

os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
default_refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
default_lora_name = 'sd_xl_offset_example-lora_1.0.safetensors'
default_lora_weight = 0.5

model_filenames = []
lora_filenames = []


def get_model_filenames(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            for ends in ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']:
                if filename.lower().endswith(ends):
                    filenames.append(filename)
                    break

    return filenames


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    return


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
    return


update_all_model_names()
