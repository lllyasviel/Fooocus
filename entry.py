import os
import math
import numpy as np
import torch
import gc
import safetensors.torch

from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


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
model.load_state_dict(safetensors.torch.load_file('./sd_xl_base_1.0.safetensors'), strict=False)

# model.conditioner.cuda()

model.conditioner.embedders[0].device = 'cpu'
model.conditioner.embedders[1].device = 'cpu'

value_dict = {
    "prompt": "a handsome man in forest", "negative_prompt": "ugly, bad", "orig_height": 1024, "orig_width": 1024,
    "crop_coords_top": 0, "crop_coords_left": 0, "target_height": 1024, "target_width": 1024, "aesthetic_score": 7.5,
    "negative_aesthetic_score": 2.0,
}

batch, batch_uc = get_batch(
    get_unique_embedder_keys_from_conditioner(model.conditioner),
    value_dict,
    [1],
)

c, uc = model.conditioner.get_unconditional_conditioning(
    batch,
    batch_uc=batch_uc)
# model.conditioner.cpu()

c = {a: b.to(torch.float16) for a, b in c.items()}
uc = {a: b.to(torch.float16) for a, b in uc.items()}

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

shape = (1, 4, 128, 128)
randn = torch.randn(shape).to(torch.float16).cuda()


def denoiser(input, sigma, c):
    return model.denoiser(model.model, input, sigma, c)


model.model.to(torch.float16).cuda()
model.denoiser.to(torch.float16).cuda()
samples_z = sampler(denoiser, randn, cond=c, uc=uc)
model.model.cpu()
model.denoiser.cpu()

a = 0
