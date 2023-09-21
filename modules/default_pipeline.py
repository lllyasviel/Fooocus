import modules.core as core
import os
import torch
import modules.path
import modules.virtual_memory as virtual_memory
import comfy.model_management

from comfy.model_base import SDXL, SDXLRefiner
from modules.patch import cfg_patched, patched_model_function
from modules.expansion import FooocusExpansion


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name):
    global xl_base, xl_base_hash, xl_base_patched, xl_base_patched_hash

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.path.modelfile_path, name)))
    model_hash = filename

    if xl_base_hash == model_hash:
        return

    if xl_base is not None:
        xl_base.to_meta()
        xl_base = None

    xl_base = core.load_model(filename)
    if not isinstance(xl_base.unet.model, SDXL):
        print('Model not supported. Fooocus only support SDXL model as the base model.')
        xl_base = None
        xl_base_hash = ''
        refresh_base_model(modules.path.default_base_model_name)
        xl_base_hash = model_hash
        xl_base_patched = xl_base
        xl_base_patched_hash = ''
        return

    xl_base_hash = model_hash
    xl_base_patched = xl_base
    xl_base_patched_hash = ''
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

    if name == 'None':
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

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

    xl_refiner_hash = model_hash
    print(f'Refiner model loaded: {model_hash}')

    xl_refiner.vae.first_stage_model.to('meta')
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
    if xl_refiner is not None:
        virtual_memory.try_move_to_virtual_memory(xl_refiner.unet.model)
        virtual_memory.try_move_to_virtual_memory(xl_refiner.clip.cond_stage_model)

    refresh_base_model(base_model_name)
    virtual_memory.load_from_virtual_memory(xl_base.unet.model)

    refresh_loras(loras)
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
def patch_all_models():
    assert xl_base is not None
    assert xl_base_patched is not None

    xl_base.unet.model_options['sampler_cfg_function'] = cfg_patched
    xl_base.unet.model_options['model_function_wrapper'] = patched_model_function

    xl_base_patched.unet.model_options['sampler_cfg_function'] = cfg_patched
    xl_base_patched.unet.model_options['model_function_wrapper'] = patched_model_function

    if xl_refiner is not None:
        xl_refiner.unet.model_options['sampler_cfg_function'] = cfg_patched
        xl_refiner.unet.model_options['model_function_wrapper'] = patched_model_function

    return


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, latent=None, denoise=1.0, tiled=False):
    patch_all_models()

    if xl_refiner is not None:
        virtual_memory.try_move_to_virtual_memory(xl_refiner.unet.model)
    virtual_memory.load_from_virtual_memory(xl_base.unet.model)

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
            callback_function=callback
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
            callback_function=callback
        )

    decoded_latent = core.decode_vae(vae=xl_base_patched.vae, latent_image=sampled_latent, tiled=tiled)
    images = core.pytorch_to_numpy(decoded_latent)

    comfy.model_management.soft_empty_cache()
    return images
