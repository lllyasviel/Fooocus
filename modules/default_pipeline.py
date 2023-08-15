import modules.core as core
import os
import torch
import modules.path

from comfy.model_base import SDXL, SDXLRefiner


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''


def refresh_base_model(name):
    global xl_base, xl_base_hash, xl_base_patched, xl_base_patched_hash
    if xl_base_hash == str(name):
        return

    filename = os.path.join(modules.path.modelfile_path, name)

    if xl_base is not None:
        xl_base.to_meta()
        xl_base = None

    xl_base = core.load_model(filename)
    if not isinstance(xl_base.unet.model, SDXL):
        print('Model not supported. Fooocus only support SDXL model as the base model.')
        xl_base = None
        xl_base_hash = ''
        refresh_base_model(modules.path.default_base_model_name)
        xl_base_hash = name
        xl_base_patched = xl_base
        xl_base_patched_hash = ''
        return

    xl_base_hash = name
    xl_base_patched = xl_base
    xl_base_patched_hash = ''
    print(f'Base model loaded: {xl_base_hash}')

    return


def refresh_refiner_model(name):
    global xl_refiner, xl_refiner_hash
    if xl_refiner_hash == str(name):
        return

    if name == 'None':
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

    filename = os.path.join(modules.path.modelfile_path, name)

    if xl_refiner is not None:
        xl_refiner.to_meta()
        xl_refiner = None

    xl_refiner = core.load_model(filename)
    if not isinstance(xl_refiner.unet.model, SDXLRefiner):
        print('Model not supported. Fooocus only support SDXL refiner as the refiner.')
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

    xl_refiner_hash = name
    print(f'Refiner model loaded: {xl_refiner_hash}')

    xl_refiner.vae.first_stage_model.to('meta')
    xl_refiner.vae = None
    return


def refresh_loras(loras):
    global xl_base, xl_base_patched, xl_base_patched_hash
    if xl_base_patched_hash == str(loras):
        return

    model = xl_base
    for name, weight in loras:
        if name == 'None':
            continue

        filename = os.path.join(modules.path.lorafile_path, name)
        model = core.load_lora(model, filename, strength_model=weight, strength_clip=weight)
    xl_base_patched = model
    xl_base_patched_hash = str(loras)
    print(f'LoRAs loaded: {xl_base_patched_hash}')

    return


refresh_base_model(modules.path.default_base_model_name)
refresh_refiner_model(modules.path.default_refiner_model_name)
refresh_loras([(modules.path.default_lora_name, 0.5), ('None', 0.5), ('None', 0.5), ('None', 0.5), ('None', 0.5)])

positive_conditions_cache = None
negative_conditions_cache = None
positive_conditions_refiner_cache = None
negative_conditions_refiner_cache = None


def clean_prompt_cond_caches():
    global positive_conditions_cache, negative_conditions_cache, \
        positive_conditions_refiner_cache, negative_conditions_refiner_cache
    positive_conditions_cache = None
    negative_conditions_cache = None
    positive_conditions_refiner_cache = None
    negative_conditions_refiner_cache = None
    return


@torch.no_grad()
def process(positive_prompt, negative_prompt, steps, switch, width, height, image_seed, sampler_name, callback):
    global positive_conditions_cache, negative_conditions_cache, \
        positive_conditions_refiner_cache, negative_conditions_refiner_cache

    positive_conditions = core.encode_prompt_condition(clip=xl_base_patched.clip, prompt=positive_prompt) if positive_conditions_cache is None else positive_conditions_cache
    negative_conditions = core.encode_prompt_condition(clip=xl_base_patched.clip, prompt=negative_prompt) if negative_conditions_cache is None else negative_conditions_cache

    positive_conditions_cache = positive_conditions
    negative_conditions_cache = negative_conditions

    empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)

    if xl_refiner is not None:
        positive_conditions_refiner = core.encode_prompt_condition(clip=xl_refiner.clip, prompt=positive_prompt) if positive_conditions_refiner_cache is None else positive_conditions_refiner_cache
        negative_conditions_refiner = core.encode_prompt_condition(clip=xl_refiner.clip, prompt=negative_prompt) if negative_conditions_refiner_cache is None else negative_conditions_refiner_cache

        positive_conditions_refiner_cache = positive_conditions_refiner
        negative_conditions_refiner_cache = negative_conditions_refiner

        sampled_latent = core.ksampler_with_refiner(
            model=xl_base_patched.unet,
            positive=positive_conditions,
            negative=negative_conditions,
            refiner=xl_refiner.unet,
            refiner_positive=positive_conditions_refiner,
            refiner_negative=negative_conditions_refiner,
            refiner_switch_step=switch,
            latent=empty_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            sampler_name=sampler_name,
            callback_function=callback
        )

    else:
        sampled_latent = core.ksampler(
            model=xl_base_patched.unet,
            positive=positive_conditions,
            negative=negative_conditions,
            latent=empty_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            sampler_name=sampler_name,
            callback_function=callback
        )

    decoded_latent = core.decode_vae(vae=xl_base_patched.vae, latent_image=sampled_latent)

    images = core.image_to_numpy(decoded_latent)

    return images
