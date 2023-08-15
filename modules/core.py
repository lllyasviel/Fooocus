import os
import random
import einops
import torch
import numpy as np

import comfy.model_management
import comfy.utils

from comfy.sd import load_checkpoint_guess_config
from nodes import VAEDecode, EmptyLatentImage, CLIPTextEncode
from comfy.sample import prepare_mask, broadcast_cond, load_additional_models, cleanup_additional_models
from modules.samplers_advanced import KSampler, KSamplerWithRefiner
from modules.patch import patch_all


patch_all()
opCLIPTextEncode = CLIPTextEncode()
opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()


class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision

    def to_meta(self):
        if self.unet is not None:
            self.unet.model.to('meta')
        if self.clip is not None:
            self.clip.cond_stage_model.to('meta')
        if self.vae is not None:
            self.vae.first_stage_model.to('meta')


@torch.no_grad()
def load_model(ckpt_filename):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)


@torch.no_grad()
def load_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora = comfy.utils.load_torch_file(lora_filename, safe_load=True)
    unet, clip = comfy.sd.load_lora_for_models(model.unet, model.clip, lora, strength_model, strength_clip)
    return StableDiffusionModel(unet=unet, clip=clip, vae=model.vae, clip_vision=model.clip_vision)


@torch.no_grad()
def encode_prompt_condition(clip, prompt):
    return opCLIPTextEncode.encode(clip=clip, text=prompt)[0]


@torch.no_grad()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.no_grad()
def decode_vae(vae, latent_image):
    return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


def get_previewer(device, latent_format):
    from latent_preview import TAESD, TAESDPreviewerImpl
    taesd_decoder_path = os.path.abspath(os.path.realpath(os.path.join("models", "vae_approx",
                                                                       latent_format.taesd_decoder_name)))

    if not os.path.exists(taesd_decoder_path):
        print(f"Warning: TAESD previews enabled, but could not find {taesd_decoder_path}")
        return None

    taesd = TAESD(None, taesd_decoder_path).to(device)

    def preview_function(x0, step, total_steps):
        global cv2_is_top
        with torch.no_grad():
            x_sample = taesd.decoder(torch.nn.functional.avg_pool2d(x0, kernel_size=(2, 2))).detach() * 255.0
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample[0]

    taesd.preview = preview_function

    return taesd


@torch.no_grad()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(1, 2 ** 64)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        y = None
        if previewer and step % 3 == 0:
            y = previewer.preview(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)
        pbar.update_absolute(step + 1, total_steps, None)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    comfy.model_management.load_model_gpu(model)
    real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler,
                       denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return out


@torch.no_grad()
def ksampler_with_refiner(model, positive, negative, refiner, refiner_positive, refiner_negative, latent,
                          seed=None, steps=30, refiner_switch_step=20, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
                          scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                          force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(1, 2 ** 64)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        y = None
        if previewer and step % 3 == 0:
            y = previewer.preview(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)
        pbar.update_absolute(step + 1, total_steps, None)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    comfy.model_management.load_model_gpu(model)

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    refiner_positive_copy = broadcast_cond(refiner_positive, noise.shape[0], device)
    refiner_negative_copy = broadcast_cond(refiner_negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = KSamplerWithRefiner(model=model, refiner_model=refiner, steps=steps, device=device,
                                  sampler=sampler_name, scheduler=scheduler,
                                  denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, refiner_positive=refiner_positive_copy,
                             refiner_negative=refiner_negative_copy, refiner_switch_step=refiner_switch_step,
                             cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback_function=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return out


@torch.no_grad()
def image_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]
