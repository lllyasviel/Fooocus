import modules.core as core
import os
import torch

from modules.path import modelfile_path, lorafile_path


xl_base_filename = os.path.join(modelfile_path, 'sd_xl_base_1.0.safetensors')
xl_refiner_filename = os.path.join(modelfile_path, 'sd_xl_refiner_1.0.safetensors')
xl_base_offset_lora_filename = os.path.join(lorafile_path, 'sd_xl_offset_example-lora_1.0.safetensors')

xl_base = core.load_model(xl_base_filename)
xl_base = core.load_lora(xl_base, xl_base_offset_lora_filename, strength_model=0.5, strength_clip=0.0)
del xl_base.vae

xl_refiner = core.load_model(xl_refiner_filename)


@torch.no_grad()
def process(positive_prompt, negative_prompt, steps, switch, width, height, image_seed, callback):
    positive_conditions = core.encode_prompt_condition(clip=xl_base.clip, prompt=positive_prompt)
    negative_conditions = core.encode_prompt_condition(clip=xl_base.clip, prompt=negative_prompt)

    positive_conditions_refiner = core.encode_prompt_condition(clip=xl_refiner.clip, prompt=positive_prompt)
    negative_conditions_refiner = core.encode_prompt_condition(clip=xl_refiner.clip, prompt=negative_prompt)

    empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)

    sampled_latent = core.ksampler_with_refiner(
        model=xl_base.unet,
        positive=positive_conditions,
        negative=negative_conditions,
        refiner=xl_refiner.unet,
        refiner_positive=positive_conditions_refiner,
        refiner_negative=negative_conditions_refiner,
        refiner_switch_step=switch,
        latent=empty_latent,
        steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
        seed=image_seed,
        callback_function=callback
    )

    decoded_latent = core.decode_vae(vae=xl_refiner.vae, latent_image=sampled_latent)

    images = core.image_to_numpy(decoded_latent)

    return images
