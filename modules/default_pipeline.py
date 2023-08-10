import os
import torch
import modules.core as core

from modules.path import modelfile_path


xl_base_filename = os.path.join(modelfile_path, 'sd_xl_base_1.0.safetensors')
xl_refiner_filename = os.path.join(modelfile_path, 'sd_xl_refiner_1.0.safetensors')

xl_base = core.load_model(xl_base_filename)


@torch.no_grad()
def process(positive_prompt, negative_prompt, width=1024, height=1024, batch_size=1):
    positive_conditions = core.encode_prompt_condition(clip=xl_base.clip, prompt=positive_prompt)
    negative_conditions = core.encode_prompt_condition(clip=xl_base.clip, prompt=negative_prompt)

    empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=batch_size)

    sampled_latent = core.ksample(
        unet=xl_base.unet,
        positive_condition=positive_conditions,
        negative_condition=negative_conditions,
        latent_image=empty_latent
    )

    decoded_latent = core.decode_vae(vae=xl_base.vae, latent_image=sampled_latent)

    images = core.image_to_numpy(decoded_latent)
    return images
