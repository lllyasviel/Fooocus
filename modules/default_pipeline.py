import modules.core as core
import os
import torch
import modules.path
import comfy.model_management

from comfy.model_patcher import ModelPatcher
from comfy.model_base import SDXL, SDXLRefiner
from modules.expansion import FooocusExpansion


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''

xl_refiner: ModelPatcher = None
xl_refiner_hash = ''

final_expansion = None
final_unet = None
final_clip = None
final_vae = None
final_refiner = None

loaded_ControlNets = {}


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global loaded_ControlNets
    cache = {}
    for p in model_paths:
        if p is not None:
            if p in loaded_ControlNets:
                cache[p] = loaded_ControlNets[p]
            else:
                cache[p] = core.load_controlnet(p)
    loaded_ControlNets = cache
    return


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
        if not isinstance(xl_refiner.model, SDXLRefiner):
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

    xl_refiner = core.load_unet_only(filename)
    xl_refiner_hash = model_hash
    print(f'Refiner model loaded: {model_hash}')
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
def clip_encode(texts, pool_top_k=1):
    global final_clip

    if final_clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(final_clip, text)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def clear_all_caches():
    xl_base.clip.fcs_cond_cache = {}
    xl_base_patched.clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        pass
    assert_model_integrity()
    comfy.model_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras):
    global final_unet, final_clip, final_vae, final_refiner, final_expansion

    refresh_refiner_model(refiner_model_name)
    refresh_base_model(base_model_name)
    refresh_loras(loras)
    assert_model_integrity()

    final_unet, final_clip, final_vae, final_refiner = \
        xl_base_patched.unet, xl_base_patched.clip, xl_base_patched.vae, xl_refiner

    if final_expansion is None:
        final_expansion = FooocusExpansion()

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=modules.path.default_refiner_model_name,
    base_model_name=modules.path.default_base_model_name,
    loras=[
        (modules.path.default_lora_name, modules.path.default_lora_weight),
        ('None', modules.path.default_lora_weight),
        ('None', modules.path.default_lora_weight),
        ('None', modules.path.default_lora_weight),
        ('None', modules.path.default_lora_weight)
    ]
)


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0):
    if latent is None:
        empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        empty_latent = latent

    sampled_latent = core.ksampler(
        model=final_unet,
        refiner=final_refiner,
        positive=positive_cond,
        negative=negative_cond,
        latent=empty_latent,
        steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
        seed=image_seed,
        denoise=denoise,
        callback_function=callback,
        cfg=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler_name,
        refiner_switch=switch
    )

    decoded_latent = core.decode_vae(vae=final_vae, latent_image=sampled_latent, tiled=tiled)
    images = core.pytorch_to_numpy(decoded_latent)

    comfy.model_management.soft_empty_cache()
    return images
