from modules.patch import patch_all

patch_all()


import os
import random
import einops
import torch
import numpy as np

import comfy.model_management
import comfy.model_detection
import comfy.model_patcher
import comfy.utils
import comfy.controlnet
import modules.sample_hijack
import comfy.samplers

from comfy.sd import load_checkpoint_guess_config
from nodes import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, VAEEncodeForInpaint, \
    ControlNetApplyAdvanced
from comfy.sample import prepare_mask
from modules.patch import patched_sampler_cfg_function, patched_model_function_wrapper
from comfy.lora import model_lora_keys_unet, model_lora_keys_clip, load_lora


opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opVAEEncodeForInpaint = VAEEncodeForInpaint()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()


class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return comfy.controlnet.load_controlnet(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
        image=image, strength=strength, start_percent=start_percent, end_percent=end_percent)


@torch.no_grad()
@torch.inference_mode()
def load_unet_only(unet_path):
    sd_raw = comfy.utils.load_torch_file(unet_path)
    sd = {}
    flag = 'model.diffusion_model.'
    for k in list(sd_raw.keys()):
        if k.startswith(flag):
            sd[k[len(flag):]] = sd_raw[k]
        del sd_raw[k]

    parameters = comfy.utils.calculate_parameters(sd)
    fp16 = comfy.model_management.should_use_fp16(model_params=parameters)
    if "input_blocks.0.0.weight" in sd:
        # ldm
        model_config = comfy.model_detection.model_config_from_unet(sd, "", fp16)
        if model_config is None:
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        new_sd = sd
    else:
        # diffusers
        model_config = comfy.model_detection.model_config_from_diffusers_unet(sd, fp16)
        if model_config is None:
            print("ERROR UNSUPPORTED UNET", unet_path)
            return None

        diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)
            else:
                print(diffusers_keys[k], k)
    offload_device = comfy.model_management.unet_offload_device()
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    return comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=offload_device)


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename)
    unet.model_options['sampler_cfg_function'] = patched_sampler_cfg_function
    unet.model_options['model_function_wrapper'] = patched_model_function_wrapper
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)


@torch.no_grad()
@torch.inference_mode()
def load_sd_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora = comfy.utils.load_torch_file(lora_filename, safe_load=False)

    if lora_filename.lower().endswith('.fooocus.patch'):
        loaded = lora
    else:
        key_map = model_lora_keys_unet(model.unet.model)
        key_map = model_lora_keys_clip(model.clip.cond_stage_model, key_map)
        loaded = load_lora(lora, key_map)

    new_unet = model.unet.clone()
    loaded_unet_keys = new_unet.add_patches(loaded, strength_model)

    new_clip = model.clip.clone()
    loaded_clip_keys = new_clip.add_patches(loaded, strength_clip)

    loaded_keys = set(list(loaded_unet_keys) + list(loaded_clip_keys))

    for x in loaded:
        if x not in loaded_keys:
            print("Lora key not loaded: ", x)

    return StableDiffusionModel(unet=new_unet, clip=new_clip, vae=model.vae, clip_vision=model.clip_vision)


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    if tiled:
        return opVAEDecodeTiled.decode(samples=latent_image, vae=vae, tile_size=512)[0]
    else:
        return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False):
    if tiled:
        return opVAEEncodeTiled.encode(pixels=pixels, vae=vae, tile_size=512)[0]
    else:
        return opVAEEncode.encode(pixels=pixels, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    return opVAEEncodeForInpaint.encode(pixels=pixels, vae=vae, mask=mask)[0]


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


VAE_approx_model = None


@torch.no_grad()
@torch.inference_mode()
def get_previewer():
    global VAE_approx_model

    if VAE_approx_model is None:
        from modules.path import vae_approx_path
        vae_approx_filename = os.path.join(vae_approx_path, 'xlvaeapp.pth')
        sd = torch.load(vae_approx_filename, map_location='cpu')
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        if comfy.model_management.should_use_fp16():
            VAE_approx_model.half()
            VAE_approx_model.current_type = torch.float16
        else:
            VAE_approx_model.float()
            VAE_approx_model.current_type = torch.float32

        VAE_approx_model.to(comfy.model_management.get_torch_device())

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            x_sample = x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample

    return preview_function


@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_fooocus_2m_sde_inpaint_seamless',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None, refiner=None, refiner_switch=-1):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer()

    def callback(step, x0, x, total_steps):
        comfy.model_management.throw_exception_if_processing_interrupted()
        y = None
        if previewer is not None:
            y = previewer(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)

    disable_pbar = False
    modules.sample_hijack.current_refiner = refiner
    modules.sample_hijack.refiner_switch_step = refiner_switch
    comfy.samplers.sample = modules.sample_hijack.sample_hacked

    try:
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                      last_step=last_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                      disable_pbar=disable_pbar, seed=seed)

        out = latent.copy()
        out["samples"] = samples
    finally:
        modules.sample_hijack.current_refiner = None

    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
