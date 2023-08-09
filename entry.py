import os
import torch
import safetensors.torch

from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

config_path = './sd_xl_base.yaml'
config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model).cpu()
model.eval()

sd = safetensors.torch.load_file('./sd_xl_base_1.0.safetensors')
model.load_state_dict(sd, strict=False)

a = 0
