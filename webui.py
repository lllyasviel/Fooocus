import os
import random
import torch
import numpy as np
import modules.core as core

from modules.path import modelfile_path


xl_base_filename = os.path.join(modelfile_path, 'sd_xl_base_1.0.safetensors')
xl_refiner_filename = os.path.join(modelfile_path, 'sd_xl_refiner_1.0.safetensors')

xl_base = core.load_model(xl_base_filename)

positive_conditions = core.encode_prompt_condition(clip=xl_base.clip, prompt='a handsome man in forest')
negative_conditions = core.encode_prompt_condition(clip=xl_base.clip, prompt='bad, ugly')

empty_latent = core.generate_empty_latent(width=1024, height=1024, batch_size=1)

sampled_latent = core.ksample(
    unet=xl_base.unet,
    positive_condition=positive_conditions,
    negative_condition=negative_conditions,
    latent_image=empty_latent
)

decoded_latent = core.decode_vae(vae=xl_base.vae, latent_image=sampled_latent)

images = core.image_to_numpy(decoded_latent)

for image in images:
    import cv2
    cv2.imwrite('a.png', image[:, :, ::-1])
