import os
from comfy.sd import load_checkpoint_guess_config

from nodes import (
    VAEDecode,
    KSamplerAdvanced,
    EmptyLatentImage,
    SaveImage,
    CLIPTextEncode,
)

from modules.path import modelfile_path


xl_base_filename = os.path.join(modelfile_path, 'sd_xl_base_1.0.safetensors')
xl_refiner_filename = os.path.join(modelfile_path, 'sd_xl_refiner_1.0.safetensors')

xl_base, xl_base_clip, xl_base_vae, xl_base_clipvision = load_checkpoint_guess_config(xl_base_filename)
del xl_base_clipvision

a = 0
