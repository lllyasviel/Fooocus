import modules.core as core
import os
import torch
import modules.path
import comfy.model_management
import comfy.latent_formats
import modules.inpaint_worker

from comfy.model_base import SDXL, SDXLRefiner
from modules.expansion import FooocusExpansion
from modules.sample_hijack import clip_separate


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ''

final_expansion = None
final_unet = None
final_clip = None
final_vae = None
final_refiner_unet = None
final_refiner_vae = None

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
        if xl_refiner.unet is None or xl_refiner.unet.model is None:
            error_message = 'You have selected an invalid refiner!'
        # elif not isinstance(xl_refiner.unet.model, SDXL) and not isinstance(xl_refiner.unet.model, SDXLRefiner):
        #     error_message = 'SD1.5 or 2.1 as refiner is not supported!'

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

    if isinstance(xl_refiner.unet.model, SDXL):
        xl_refiner.clip = None
        xl_refiner.vae = None
    elif isinstance(xl_refiner.unet.model, SDXLRefiner):
        xl_refiner.clip = None
        xl_refiner.vae = None
    else:
        xl_refiner.clip = None

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
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion

    refresh_refiner_model(refiner_model_name)
    refresh_base_model(base_model_name)
    refresh_loras(loras)
    assert_model_integrity()

    final_unet = xl_base_patched.unet
    final_clip = xl_base_patched.clip
    final_vae = xl_base_patched.vae

    final_unet.model.diffusion_model.in_inpaint = False

    if xl_refiner is None:
        final_refiner_unet = None
        final_refiner_vae = None
    else:
        final_refiner_unet = xl_refiner.unet
        final_refiner_unet.model.diffusion_model.in_inpaint = False

        final_refiner_vae = xl_refiner.vae

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
def vae_parse(x, tiled=False, use_interpose=True):
    if final_vae is None or final_refiner_vae is None:
        return x

    if use_interpose:
        print('VAE interposing ...')
        import fooocus_extras.vae_interpose
        x = fooocus_extras.vae_interpose.parse(x)
        print('VAE interposed ...')
    else:
        print('VAE parsing ...')
        x = core.decode_vae(vae=final_vae, latent_image=x, tiled=tiled)
        x = core.encode_vae(vae=final_refiner_vae, pixels=x, tiled=tiled)
        print('VAE parsed ...')

    return x


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_all(sampler, model, scheduler, steps):
    from comfy.samplers import calculate_sigmas_scheduler

    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = calculate_sigmas_scheduler(model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas(sampler, model, scheduler, steps, denoise):
    if denoise is None or denoise > 0.9999:
        sigmas = calculate_sigmas_all(sampler, model, scheduler, steps)
    else:
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_all(sampler, model, scheduler, new_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint'):
    assert refiner_swap_method in ['joint', 'separate', 'vae', 'upscale']

    if final_refiner_unet is not None:
        if isinstance(final_refiner_unet.model.latent_format, comfy.latent_formats.SD15) \
                and refiner_swap_method != 'upscale':
            refiner_swap_method = 'vae'

    print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

    if latent is None:
        empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        empty_latent = latent

    if refiner_swap_method == 'joint':
        sampled_latent = core.ksampler(
            model=final_unet,
            refiner=final_refiner_unet,
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
            refiner_switch=switch,
            previewer_start=0,
            previewer_end=steps,
        )
        decoded_latent = core.decode_vae(vae=final_vae, latent_image=sampled_latent, tiled=tiled)
        images = core.pytorch_to_numpy(decoded_latent)
        return images

    if refiner_swap_method == 'upscale':
        target_model = final_refiner_unet
        if target_model is None:
            target_model = final_unet

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=final_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=final_clip),
            latent=empty_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
        )

        target_model = final_refiner_vae
        if target_model is None:
            target_model = final_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)
        images = core.pytorch_to_numpy(decoded_latent)
        return images

    if refiner_swap_method == 'separate':
        sampled_latent = core.ksampler(
            model=final_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=empty_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=False,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
        )
        print('Refiner swapped by changing ksampler. Noise preserved.')

        target_model = final_refiner_unet
        if target_model is None:
            target_model = final_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=final_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=final_clip),
            latent=sampled_latent,
            steps=steps, start_step=switch, last_step=steps, disable_noise=True, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
        )

        target_model = final_refiner_vae
        if target_model is None:
            target_model = final_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)
        images = core.pytorch_to_numpy(decoded_latent)
        return images

    if refiner_swap_method == 'vae':
        sigmas = calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=final_unet.model, steps=steps, denoise=denoise)
        sigmas_a = sigmas[:switch]
        sigmas_b = sigmas[switch:]

        if final_refiner_unet is not None:
            k1 = final_refiner_unet.model.latent_format.scale_factor
            k2 = final_unet.model.latent_format.scale_factor
            k = float(k1) / float(k2)
            sigmas_b = sigmas_b * k

        sigmas = torch.cat([sigmas_a, sigmas_b], dim=0)

        sampled_latent = core.ksampler(
            model=final_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=empty_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            sigmas=sigmas
        )
        print('Fooocus VAE-based swap.')

        target_model = final_refiner_unet
        if target_model is None:
            target_model = final_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = vae_parse(sampled_latent)

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.swap()

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=final_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=final_clip),
            latent=sampled_latent,
            steps=steps, start_step=switch, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            sigmas=sigmas
        )

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.swap()

        target_model = final_refiner_vae
        if target_model is None:
            target_model = final_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)
        images = core.pytorch_to_numpy(decoded_latent)
        return images
