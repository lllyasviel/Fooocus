import modules.core as core
import os
import torch
import modules.patch
import modules.config
import ldm_patched.modules.model_management
import ldm_patched.modules.latent_formats
import modules.inpaint_worker
import extras.vae_interpose as vae_interpose
from extras.expansion import FooocusExpansion

from ldm_patched.modules.model_base import SDXL, SDXLRefiner
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
def synthesize_refiner_model():
    global model_base, model_refiner

    print('Synthetic Refiner Activated')
    model_refiner = core.StableDiffusionModel(
        unet=model_base.unet,
        vae=model_base.vae,
        clip=model_base.clip,
        clip_vision=model_base.clip_vision,
        filename=model_base.filename
    )
    model_refiner.vae = None
    model_refiner.clip = None
    model_refiner.clip_vision = None

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
    ldm_patched.modules.model_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras,
                       base_model_additional_loras=None, use_synthetic_refiner=False):
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion

    final_unet = None
    final_clip = None
    final_vae = None
    final_refiner_unet = None
    final_refiner_vae = None

    if use_synthetic_refiner and refiner_model_name == 'None':
        print('Synthetic Refiner Activated')
        refresh_base_model(base_model_name)
        synthesize_refiner_model()
    else:
        refresh_refiner_model(refiner_model_name)
        refresh_base_model(base_model_name)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    final_refiner_unet = model_refiner.unet_with_lora
    final_refiner_vae = model_refiner.vae

    if final_expansion is None:
        final_expansion = FooocusExpansion()

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=modules.config.default_refiner_model_name,
    base_model_name=modules.config.default_base_model_name,
    loras=modules.config.default_loras
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
    from ldm_patched.modules.samplers import calculate_sigmas_scheduler

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
def get_candidate_vae(steps, switch, denoise=1.0, refiner_swap_method='joint'):
    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if final_refiner_vae is not None and final_refiner_unet is not None:
        if denoise > 0.9:
            return final_vae, final_refiner_vae
        else:
            if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                return final_vae, None
            else:
                return final_refiner_vae, None

    return final_vae, final_refiner_vae


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint', disable_preview=False):
    target_unet, target_vae, target_refiner_unet, target_refiner_vae, target_clip \
        = final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip

    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if final_refiner_vae is not None and final_refiner_unet is not None:
        # Refiner Use Different VAE (then it is SD15)
        if denoise > 0.9:
            refiner_swap_method = 'vae'
        else:
            refiner_swap_method = 'joint'
            if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                    = final_unet, final_vae, None, None
                print(f'[Sampler] only use Base because of partial denoise.')
            else:
                positive_cond = clip_separate(positive_cond, target_model=final_refiner_unet.model, target_clip=final_clip)
                negative_cond = clip_separate(negative_cond, target_model=final_refiner_unet.model, target_clip=final_clip)
                target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                    = final_refiner_unet, final_refiner_vae, None, None
                print(f'[Sampler] only use Refiner because of partial denoise.')

    print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

    if latent is None:
        initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        initial_latent = latent

    minmax_sigmas = calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=final_unet.model, steps=steps, denoise=denoise)
    sigma_min, sigma_max = minmax_sigmas[minmax_sigmas > 0].min(), minmax_sigmas.max()
    sigma_min = float(sigma_min.cpu().numpy())
    sigma_max = float(sigma_max.cpu().numpy())
    print(f'[Sampler] sigma_min = {sigma_min}, sigma_max = {sigma_max}')

    modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
        initial_latent['samples'].to(ldm_patched.modules.model_management.get_torch_device()),
        sigma_min, sigma_max, seed=image_seed, cpu=False)

    decoded_latent = None

    if refiner_swap_method == 'joint':
        sampled_latent = core.ksampler(
            model=target_unet,
            refiner=target_refiner_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
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
            disable_preview=disable_preview
        )
        decoded_latent = core.decode_vae(vae=target_vae, latent_image=sampled_latent, tiled=tiled)

    if refiner_swap_method == 'separate':
        sampled_latent = core.ksampler(
            model=target_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=False,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview
        )
        print('Refiner swapped by changing ksampler. Noise preserved.')

        target_model = target_refiner_unet
        if target_model is None:
            target_model = target_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
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
            disable_preview=disable_preview
        )

        target_model = target_refiner_vae
        if target_model is None:
            target_model = target_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

    if refiner_swap_method == 'vae':
        modules.patch.patch_settings[os.getpid()].eps_record = 'vae'

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.unswap()

        sampled_latent = core.ksampler(
            model=target_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview
        )
        print('Fooocus VAE-based swap.')

        target_model = target_refiner_unet
        if target_model is None:
            target_model = target_unet
            print('Use base model to refine itself - this may because of developer mode.')

        sampled_latent = vae_parse(sampled_latent)

        k_sigmas = 1.4
        sigmas = calculate_sigmas(sampler=sampler_name,
                                  scheduler=scheduler_name,
                                  model=target_model.model,
                                  steps=steps,
                                  denoise=denoise)[switch:] * k_sigmas
        len_sigmas = len(sigmas) - 1

        noise_mean = torch.mean(modules.patch.patch_settings[os.getpid()].eps_record, dim=1, keepdim=True)

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.swap()

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
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
            noise_mean=noise_mean,
            disable_preview=disable_preview
        )

        target_model = target_refiner_vae
        if target_model is None:
            target_model = target_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)

    images = core.pytorch_to_numpy(decoded_latent)
    modules.patch.patch_settings[os.getpid()].eps_record = None
    return images
