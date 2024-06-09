from PIL import Image
import numpy as np
import torch
from rembg import remove, new_session
from extras.GroundingDINO.util.inference import default_groundingdino

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_grounded_sam(input_image, text_prompt, box_threshold, text_threshold):

    # run grounding dino model
    detections, _, _, _ = default_groundingdino(
        image=np.array(input_image),
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return detections.xyxy


def generate_mask_from_image(image, mask_model, extras, box_erode_or_dilate: int=0):
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    if mask_model == 'sam':
        img = Image.fromarray(image)
        boxes = run_grounded_sam(img, extras['sam_prompt_text'], box_threshold=extras['box_threshold'], text_threshold=extras['text_threshold'])
        # use full image if no box has been found
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]]) if len(boxes) == 0 else boxes

        extras['sam_prompt'] = []
        # from PIL import ImageDraw
        # draw = ImageDraw.Draw(img)
        for idx, box in enumerate(boxes):
            box_list = box.tolist()
            if box_erode_or_dilate != 0:
                box_list[0] -= box_erode_or_dilate
                box_list[1] -= box_erode_or_dilate
                box_list[2] += box_erode_or_dilate
                box_list[3] += box_erode_or_dilate
        #     draw.rectangle(box_list, fill=128, outline ="red")
            extras['sam_prompt'] += [{"type": "rectangle", "data": box_list}]
        # img.show()

    return remove(
        image,
        session=new_session(mask_model, **extras),
        only_mask=True,
        # post_process_mask=True,
        **extras
    )
