# modified version of https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor/blob/master/scripts/censor.py

import numpy as np
import torch
import modules.core as core

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
import modules.config

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None


def numpy_to_pil(image):
    image = (image * 255).round().astype("uint8")

    #pil_image = Image.fromarray(image, 'RGB')
    pil_image = Image.fromarray(image)

    return pil_image


# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id, cache_dir=modules.config.path_safety_checker_models)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id, cache_dir=modules.config.path_safety_checker_models)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

    return x_checked_image, has_nsfw_concept


def censor_single(x):
    x_checked_image, has_nsfw_concept = check_safety(x)

    # replace image with black pixels, keep dimensions
    # workaround due to different numpy / pytorch image matrix format
    if has_nsfw_concept[0]:
        imageshape = x_checked_image.shape
        x_checked_image = np.zeros((imageshape[0], imageshape[1], 3), dtype = np.uint8)

    return x_checked_image


def censor_batch(images):
    images = [censor_single(image) for image in images]

    return images
