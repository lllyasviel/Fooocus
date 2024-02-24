from typing import Tuple, List

import ldm_patched.modules.model_management as model_management
from ldm_patched.modules.model_patcher import ModelPatcher
from modules.config import path_inpaint
from modules.model_loader import load_file_from_url

import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model
from groundingdino.util.inference import load_model, preprocess_caption, get_phrases_from_posmap


class GroundingDinoModel(Model):
    def __init__(self):
        self.config_file = 'extras/GroundingDINO/config/GroundingDINO_SwinT_OGC.py'
        self.model = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')

    def predict_with_caption(
            self,
            image: np.ndarray,
            caption: str,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        if self.model is None:
            filename = load_file_from_url(
                url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                file_name='groundingdino_swint_ogc.pth',
                model_dir=path_inpaint)
            model = load_model(model_config_path=self.config_file, model_checkpoint_path=filename)

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()

            model.to(self.offload_device)

            self.model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

        model_management.load_model_gpu(self.model)

        processed_image = GroundingDinoModel.preprocess_image(image_bgr=image).to(self.load_device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.load_device)
        source_h, source_w, _ = image.shape
        detections = GroundingDinoModel.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    # override to use model wrapped by patcher
    model = model.model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases


default_groundingdino = GroundingDinoModel().predict_with_caption
