# Consistent with Kohya to reduce differences between model training and inference.

import torch
import math
import einops
import numpy as np

import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.modules.model_sampling
import ldm_patched.modules.sd1_clip

from ldm_patched.ldm.modules.diffusionmodules.util import make_beta_schedule


def patched_timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    # Consistent with Kohya to reduce differences between model training and inference.

    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = einops.repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def patched_register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    # Consistent with Kohya to reduce differences between model training and inference.

    if given_betas is not None:
        betas = given_betas
    else:
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)
    self.linear_start = linear_start
    self.linear_end = linear_end
    sigmas = torch.tensor(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, dtype=torch.float32)
    self.set_sigmas(sigmas)
    alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
    self.set_alphas_cumprod(alphas_cumprod)
    return


def patch_all_precision():
    ldm_patched.ldm.modules.diffusionmodules.openaimodel.timestep_embedding = patched_timestep_embedding
    ldm_patched.modules.model_sampling.ModelSamplingDiscrete._register_schedule = patched_register_schedule
    return
