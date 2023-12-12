# https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py

import os
import torch
import safetensors.torch as sf
import torch.nn as nn
import ldm_patched.modules.model_management

from ldm_patched.modules.model_patcher import ModelPatcher
from modules.config import path_vae_approx


class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.join = nn.ReLU()
        self.long = nn.Sequential(
            nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        y = self.long(x)
        z = self.join(y + x)
        return z


class Interposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.chan = 4
        self.hid = 128

        self.head_join = nn.ReLU()
        self.head_short = nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1)
        self.head_long = nn.Sequential(
            nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=1),
        )
        self.core = nn.Sequential(
            Block(self.hid),
            Block(self.hid),
            Block(self.hid),
        )
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.hid, self.chan, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.head_join(
            self.head_long(x) +
            self.head_short(x)
        )
        z = self.core(y)
        return self.tail(z)


vae_approx_model = None
vae_approx_filename = os.path.join(path_vae_approx, 'xl-to-v1_interposer-v3.1.safetensors')


def parse(x):
    global vae_approx_model

    x_origin = x.clone()

    if vae_approx_model is None:
        model = Interposer()
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
