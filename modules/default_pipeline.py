import modules.core as core
import os
import torch
import modules.path
import comfy.model_management

from comfy.model_base import SDXL, SDXLRefiner
from modules.expansion import FooocusExpansion


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''


@torch.no_grad()
@torch.inference_mode()
def assert_model_integrity():
    error_message = None

    if xl_base is None:
        error_message = 'You have not selected SDXL base model.'

    if xl_base_patched is None:
        error_message = 'You have not selected SDXL base model.'

    if not isinstance(xl_base.unet.model, SDXL):
        error_message = 'You have selected base model other than SDXL. This is not supported yet.'

    if not isinstance(xl_base_patched.unet.model, SDXL):
        error_message = 'You have selected base model other than SDXL. This is not supported yet.'

    if xl_refiner is not None:
        if not isinstance(xl_refiner.unet.model, SDXLRefiner):
            error_message = 'You have selected refiner model other than SDXL refiner. This is not supported yet.'

    if error_message is not None:
        raise NotImplementedError(error_message)

    return True


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name):
    global xl_base, xl_base_hash, xl_base_patched, xl_base_patched_hash

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.path.modelfile_path, name)))
    model_hash = filename

    if xl_base_hash == model_hash:
        return

    xl_base = None
    xl_base_hash = ''
    xl_base_patched = None
    xl_base_patched_hash = ''

    xl_base = core.load_model(filename)
    xl_base_hash = model_hash
    print(f'Base model loaded: {model_hash}')
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_refiner_model(name):
    global xl_refiner, xl_refiner_hash

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.path.modelfile_path, name)))
    model_hash = filename

    if xl_refiner_hash == model_hash:
        return

    xl_refiner = None
    xl_refiner_hash = ''

    if name == 'None':
        print(f'Refiner unloaded.')
        return

    xl_refiner = core.load_model(filename)
    xl_refiner_hash = model_hash
    print(f'Refiner model loaded: {model_hash}')

    # Remove VAE
    xl_refiner.vae = None
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras):
    global xl_base, xl_base_patched, xl_base_patched_hash
    if xl_base_patched_hash == str(loras):
        return

    model = xl_base
    for name, weight in loras:
        if name == 'None':
            continue

        if os.path.exists(name):
            filename = name
        else:
            filename = os.path.join(modules.path.lorafile_path, name)

        assert os.path.exists(filename), 'Lora file not found!'

        model = core.load_sd_lora(model, filename, strength_model=weight, strength_clip=weight)
    xl_base_patched = model
    xl_base_patched_hash = str(loras)
    print(f'LoRAs loaded: {xl_base_patched_hash}')

    return


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False):
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[text] = result
    if verbose:
        print(f'[CLIP Encoded] {text}')
    return result


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond):
    c, p = cond[0]
    c = c[..., -1280:].clone()
    p = p["pooled_output"]
    return [[c, {"pooled_output": p}]]


@torch.no_grad()
@torch.inference_mode()
def clip_encode(sd, texts, pool_top_k=1):
    if sd is None:
        return None
    if sd.clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    clip = sd.clip
    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(clip, text)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def clear_sd_cond_cache(sd):
    if sd is None:
        return None
    if sd.clip is None:
        return None
    sd.clip.fcs_cond_cache = {}
    return


@torch.no_grad()
@torch.inference_mode()
def clear_all_caches():
    clear_sd_cond_cache(xl_base_patched)
    clear_sd_cond_cache(xl_refiner)


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras):
    refresh_refiner_model(refiner_model_name)
    refresh_base_model(base_model_name)
    refresh_loras(loras)
    assert_model_integrity()
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=modules.path.default_refiner_model_name,
    base_model_name=modules.path.default_base_model_name,
    loras=[(modules.path.default_lora_name, 0.5), ('None', 0.5), ('None', 0.5), ('None', 0.5), ('None', 0.5)]
)

expansion = FooocusExpansion()


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0):
    if latent is None:
        empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        empty_latent = latent

    if xl_refiner is not None:
        sampled_latent = core.ksampler_with_refiner(
            model=xl_base_patched.unet,
            positive=positive_cond[0],
            negative=negative_cond[0],
            refiner=xl_refiner.unet,
            refiner_positive=positive_cond[1],
            refiner_negative=negative_cond[1],
            refiner_switch_step=switch,
            latent=empty_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale
        )
    else:
        sampled_latent = core.ksampler(
            model=xl_base_patched.unet,
            positive=positive_cond[0],
            negative=negative_cond[0],
            latent=empty_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale
        )

    decoded_latent = core.decode_vae(vae=xl_base_patched.vae, latent_image=sampled_latent, tiled=tiled)
    images = core.pytorch_to_numpy(decoded_latent)

    comfy.model_management.soft_empty_cache()
    return images
