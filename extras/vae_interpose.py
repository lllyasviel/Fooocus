# https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py

import os

import safetensors.torch as sf
import torch
import torch.nn as nn

import ldm_patched.modules.model_management
from ldm_patched.modules.model_patcher import ModelPatcher
from modules.config import path_vae_approx


class ResBlock(nn.Module):
    """Block with residuals"""

    def __init__(self, ch):
        super().__init__()
        self.join = nn.ReLU()
        self.norm = nn.BatchNorm2d(ch)
        self.long = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.join(self.long(x) + x)


class ExtractBlock(nn.Module):
    """Increase no. of channels by [out/in]"""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.join = nn.ReLU()
        self.short = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.long = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.join(self.long(x) + self.short(x))


class InterposerModel(nn.Module):
    """Main neural network"""

    def __init__(self, ch_in=4, ch_out=4, ch_mid=64, scale=1.0, blocks=12):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_mid = ch_mid
        self.blocks = blocks
        self.scale = scale

        self.head = ExtractBlock(self.ch_in, self.ch_mid)
        self.core = nn.Sequential(
            nn.Upsample(scale_factor=self.scale, mode="nearest"),
            *[ResBlock(self.ch_mid) for _ in range(blocks)],
            nn.BatchNorm2d(self.ch_mid),
            nn.SiLU(),
        )
        self.tail = nn.Conv2d(self.ch_mid, self.ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.head(x)
        z = self.core(y)
        return self.tail(z)


vae_approx_model = None
vae_approx_filename = os.path.join(path_vae_approx, 'xl-to-v1_interposer-v4.0.safetensors')


def parse(x):
    global vae_approx_model

    x_origin = x.clone()

    if vae_approx_model is None:
        model = InterposerModel()
        model.eval()
        sd = sf.load_file(vae_approx_filename)
        model.load_state_dict(sd)
        fp16 = ldm_patched.modules.model_management.should_use_fp16()
        if fp16:
            model = model.half()
        vae_approx_model = ModelPatcher(
            model=model,
            load_device=ldm_patched.modules.model_management.get_torch_device(),
            offload_device=torch.device('cpu')
        )
        vae_approx_model.dtype = torch.float16 if fp16 else torch.float32

    ldm_patched.modules.model_management.load_model_gpu(vae_approx_model)

    x = x_origin.to(device=vae_approx_model.load_device, dtype=vae_approx_model.dtype)
    x = vae_approx_model.model(x).to(x_origin)
    return x
