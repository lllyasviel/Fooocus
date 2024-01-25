from PIL import Image
import numpy as np
import torch
from rembg import remove, new_session
from groundingdino.util.inference import Model as GroundingDinoModel

from modules.model_loader import load_file_from_url
from modules.config import path_inpaint

config_file = 'extras/GroundingDINO/config/GroundingDINO_SwinT_OGC.py'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


groundingdino_model = None


def run_grounded_sam(input_image, text_prompt, box_threshold, text_threshold):

    global groundingdino_model

    if groundingdino_model is None:
        filename = load_file_from_url(
            url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            model_dir=path_inpaint)
        groundingdino_model = GroundingDinoModel(model_config_path=config_file, model_checkpoint_path=filename, device=device)


    # run grounding dino model
    boxes, _ = groundingdino_model.predict_with_caption(
        image=np.array(input_image),
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes.xyxy


def generate_mask_from_image(image, mask_model, extras):
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    if mask_model == 'sam':
        boxes = run_grounded_sam(Image.fromarray(image), extras['sam_prompt_text'], box_threshold=extras['box_threshold'], text_threshold=extras['text_threshold'])
        extras['sam_prompt'] = []
        for idx, box in enumerate(boxes):
            extras['sam_prompt'] += [{"type": "rectangle", "data": box.tolist()}]

    return remove(
        image,
        session=new_session(mask_model, **extras),
        only_mask=True,
        **extras
    )
