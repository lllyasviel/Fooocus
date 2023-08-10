import random
import torch
import numpy as np

from comfy.sd import load_checkpoint_guess_config
from nodes import VAEDecode, KSamplerAdvanced, EmptyLatentImage, CLIPTextEncode


opCLIPTextEncode = CLIPTextEncode()
opEmptyLatentImage = EmptyLatentImage()
opKSamplerAdvanced = KSamplerAdvanced()
opVAEDecode = VAEDecode()


class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision


@torch.no_grad()
def load_model(ckpt_filename):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)


@torch.no_grad()
def encode_prompt_condition(clip, prompt):
    return opCLIPTextEncode.encode(clip=clip, text=prompt)[0]


@torch.no_grad()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.no_grad()
def decode_vae(vae, latent_image):
    return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


@torch.no_grad()
def ksample(unet, positive_condition, negative_condition, latent_image, add_noise=True, noise_seed=None, steps=25, cfg=9,
            sampler_name='euler_ancestral', scheduler='normal', start_at_step=None, end_at_step=None,
            return_with_leftover_noise=False):
    return opKSamplerAdvanced.sample(
        add_noise='enable' if add_noise else 'disable',
        noise_seed=noise_seed if isinstance(noise_seed, int) else random.randint(1, 2 ** 64),
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        start_at_step=0 if start_at_step is None else start_at_step,
        end_at_step=steps if end_at_step is None else end_at_step,
        return_with_leftover_noise='enable' if return_with_leftover_noise else 'disable',
        model=unet,
        positive=positive_condition,
        negative=negative_condition,
        latent_image=latent_image,
    )[0]


@torch.no_grad()
def image_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]
