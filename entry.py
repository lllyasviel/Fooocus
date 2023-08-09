import os
import torch

from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

config_path = './sd_xl_base.yaml'
config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model).cpu()

a = 0
