import numpy as np
from rembg import remove, new_session
from extras.GroundingDINO.util.inference import default_groundingdino


def generate_mask_from_image(image: np.ndarray, mask_model: str, extras: dict, box_erode_or_dilate: int=0, debug_dino: bool=False) -> np.ndarray | None:
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    if mask_model == 'sam':
        detections, _, _, _ = default_groundingdino(
            image=image,
            caption=extras['sam_prompt_text'],
            box_threshold=extras['box_threshold'],
            text_threshold=extras['text_threshold']
        )
        detection_boxes = detections.xyxy
        # use full image if no box has been found
        detection_boxes = np.array([[0, 0, image.shape[1], image.shape[0]]]) if len(detection_boxes) == 0 else detection_boxes

        extras['sam_prompt'] = []
        for idx, box in enumerate(detection_boxes):
            box_list = box.tolist()
            if box_erode_or_dilate != 0:
                box_list[0] -= box_erode_or_dilate
                box_list[1] -= box_erode_or_dilate
                box_list[2] += box_erode_or_dilate
                box_list[3] += box_erode_or_dilate
            extras['sam_prompt'] += [{"type": "rectangle", "data": box_list}]

        if debug_dino:
            from PIL import ImageDraw, Image
            debug_dino_image = Image.new("RGB", (image.shape[1], image.shape[0]), color="black")
            draw = ImageDraw.Draw(debug_dino_image)
            for box in extras['sam_prompt']:
                draw.rectangle(box['data'], fill="white")
            return np.array(debug_dino_image)

    return remove(
        image,
        session=new_session(mask_model, **extras),
        only_mask=True,
        **extras
    )
