import os
import torch

from fcbh_extras.chainner_models.architecture.RRDB import RRDBNet as ESRGAN
from fcbh_extras.nodes_upscale_model import ImageUpscaleWithModel
from collections import OrderedDict
from modules.config import path_upscale_models

model_filename = os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')
opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None


def perform_upscale(img):
    global model
    if model is None:
        sd = torch.load(model_filename)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()
    return opImageUpscaleWithModel.upscale(model, img)[0]
