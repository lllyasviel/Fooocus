import modules.core as core
import os
import torch
import modules.patch
import modules.config
import fcbh.model_management
import fcbh.latent_formats
import modules.inpaint_worker
import fooocus_extras.vae_interpose as vae_interpose

from fcbh.model_base import SDXL, SDXLRefiner
from modules.expansion import FooocusExpansion
from modules.sample_hijack import clip_separate


model_base = core.StableDiffusionModel()
model_refiner = core.StableDiffusionModel()

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

    if not isinstance(model_base.unet_with_lora.model, SDXL):
        error_message = 'You have selected base model other than SDXL. This is not supported yet.'

    if error_message is not None:
        raise NotImplementedError(error_message)

    return True


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name):
    global model_base

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.config.path_checkpoints, name)))

    if model_base.filename == filename:
        return

    model_base = core.StableDiffusionModel()
    model_base = core.load_model(filename)
    print(f'Base model loaded: {model_base.filename}')
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_refiner_model(name):
    global model_refiner

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.config.path_checkpoints, name)))

    if model_refiner.filename == filename:
        return

    model_refiner = core.StableDiffusionModel()

    if name == 'None':
        print(f'Refiner unloaded.')
        return

    model_refiner = core.load_model(filename)
    print(f'Refiner model loaded: {model_refiner.filename}')

    if isinstance(model_refiner.unet.model, SDXL):
        model_refiner.clip = None
        model_refiner.vae = None
    elif isinstance(model_refiner.unet.model, SDXLRefiner):
        model_refiner.clip = None
        model_refiner.vae = None
    else:
        model_refiner.clip = None

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras, base_model_additional_loras=None):
    global model_base, model_refiner

    if not isinstance(base_model_additional_loras, list):
        base_model_additional_loras = []

    model_base.refresh_loras(loras + base_model_additional_loras)
    model_refiner.refresh_loras(loras)

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
def clone_cond(conds):
    results = []

    for c, p in conds:
        p = p["pooled_output"]

        if isinstance(c, torch.Tensor):
            c = c.clone()

        if isinstance(p, torch.Tensor):
            p = p.clone()

        results.append([c, {"pooled_output": p}])

    return results


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
    final_clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        pass
    assert_model_integrity()
    fcbh.model_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras, base_model_additional_loras=None):
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion

    final_unet = None
    final_clip = None
    final_vae = None
    final_refiner_unet = None
    final_refiner_vae = None

    refresh_refiner_model(refiner_model_name)
    refresh_base_model(base_model_name)
    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    final_unet.model.diffusion_model.in_inpaint = False

    final_refiner_unet = model_refiner.unet_with_lora
    final_refiner_vae = model_refiner.vae

    if final_refiner_unet is not None:
        final_refiner_unet.model.diffusion_model.in_inpaint = False

    if final_expansion is None:
        final_expansion = FooocusExpansion()

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=modules.config.default_refiner_model_name,
    base_model_name=modules.config.default_base_model_name,
    loras=[
        (modules.config.default_lora_name, modules.config.default_lora_weight),
        ('None', modules.config.default_lora_weight),
        ('None', modules.config.default_lora_weight),
        ('None', modules.config.default_lora_weight),
        ('None', modules.config.default_lora_weight)
    ]
)


@torch.no_grad()
@torch.inference_mode()
def vae_parse(latent):
    if final_refiner_vae is None:
        return latent

    result = vae_interpose.parse(latent["samples"])
    return {'samples': result}


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_all(sampler, model, scheduler, steps):
    from fcbh.samplers import calculate_sigmas_scheduler

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
    global final_unet, final_refiner_unet, final_vae, final_refiner_vae

    assert refiner_swap_method in ['joint', 'separate', 'vae', 'upscale']

    refiner_use_different_vae = final_refiner_vae is not None and final_refiner_unet is not None

    if refiner_swap_method == 'upscale':
        if not refiner_use_different_vae:
            refiner_swap_method = 'joint'
    else:
        if refiner_use_different_vae:
            if denoise > 0.95:
                refiner_swap_method = 'vae'
            else:
                # VAE swap only support full denoise
                # Disable refiner to avoid SD15 in joint/separate swap
                final_refiner_unet = None
                final_refiner_vae = None

    print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

    if latent is None:
        empty_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        empty_latent = latent

    minmax_sigmas = calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=final_unet.model, steps=steps, denoise=denoise)
    sigma_min, sigma_max = minmax_sigmas[minmax_sigmas > 0].min(), minmax_sigmas.max()
    sigma_min = float(sigma_min.cpu().numpy())
    sigma_max = float(sigma_max.cpu().numpy())
    print(f'[Sampler] sigma_min = {sigma_min}, sigma_max = {sigma_max}')

    modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
        empty_latent['samples'].to(fcbh.model_management.get_torch_device()),
        sigma_min, sigma_max, seed=image_seed, cpu=False)

    decoded_latent = None

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

    if refiner_swap_method == 'upscale':
        sampled_latent = core.ksampler(
            model=final_refiner_unet,
            positive=clip_separate(positive_cond, target_model=final_refiner_unet.model, target_clip=final_clip),
            negative=clip_separate(negative_cond, target_model=final_refiner_unet.model, target_clip=final_clip),
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
        decoded_latent = core.decode_vae(vae=final_refiner_vae, latent_image=sampled_latent, tiled=tiled)

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

    if refiner_swap_method == 'vae':
        modules.patch.eps_record = 'vae'

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.unswap()

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
            previewer_end=steps
        )
        print('Fooocus VAE-based swap.')

        target_model = final_refiner_unet
        if target_model is None:
            target_model = final_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = vae_parse(sampled_latent)

        k_sigmas = 1.4
        sigmas = calculate_sigmas(sampler=sampler_name,
                                  scheduler=scheduler_name,
                                  model=target_model.model,
                                  steps=steps,
                                  denoise=denoise)[switch:] * k_sigmas
        len_sigmas = len(sigmas) - 1

        noise_mean = torch.mean(modules.patch.eps_record, dim=1, keepdim=True)

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.swap()

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=final_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=final_clip),
            latent=sampled_latent,
            steps=len_sigmas, start_step=0, last_step=len_sigmas, disable_noise=False, force_full_denoise=True,
            seed=image_seed+1,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            sigmas=sigmas,
            noise_mean=noise_mean
        )

        target_model = final_refiner_vae
        if target_model is None:
            target_model = final_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

    images = core.pytorch_to_numpy(decoded_latent)
    modules.patch.eps_record = None
    return images
