import os
import torch
import safetensors.torch

from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler

sampler = EulerAncestralSampler(
    num_steps=40,
    discretization_config={
        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    },
    guider_config={
        "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
        "params": {"scale": 9.0, "dyn_thresh_config": {
            "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
        }},
    },
    eta=1.0,
    s_noise=1.0,
    verbose=True,
)

config_path = './sd_xl_base.yaml'
config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model).cpu()
model.eval()

sd = safetensors.torch.load_file('./sd_xl_base_1.0.safetensors')
model.load_state_dict(sd, strict=False)

a = 0
