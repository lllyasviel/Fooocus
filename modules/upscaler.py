import os
import torch
import modules.core as core

from fcbh_extras.chainner_models.architecture.RRDB import RRDBNet as ESRGAN
from fcbh_extras.nodes_upscale_model import ImageUpscaleWithModel
from collections import OrderedDict
from modules.config import path_upscale_models

model_filename = os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')
opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None


def perform_upscale(img):
    global model

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if model is None:
        sd = torch.load(model_filename)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    return img
