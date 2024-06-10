import numpy as np
import torch
from rembg import remove, new_session
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import remove_small_regions

from extras.GroundingDINO.util.inference import default_groundingdino
import modules.config


class SAMOptions:
    def __init__(self,
                 # GroundingDINO
                 dino_prompt: str = '',
                 dino_box_threshold=0.3,
                 dino_text_threshold=0.25,
                 box_erode_or_dilate=0,

                 # SAM
                 max_num_boxes=2,
                 model_type="vit_l"
                 ):
        self.dino_prompt = dino_prompt
        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold
        self.box_erode_or_dilate = box_erode_or_dilate
        self.max_num_boxes = max_num_boxes
        self.model_type = model_type


def optimize_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    removes small disconnected regions and holes
    """
    fine_masks = []
    for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
        fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
    masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
    return torch.from_numpy(masks)


def generate_mask_from_image(image: np.ndarray, mask_model: str = 'sam', extras=None,
                             sam_options: SAMOptions | None = SAMOptions) -> np.ndarray | None:
    if image is None:
        return

    if extras is None:
        extras = {}

    if 'image' in image:
        image = image['image']

    if mask_model != 'sam' and sam_options is None:
        return remove(
            image,
            session=new_session(mask_model, **extras),
            only_mask=True,
            **extras
        )

    assert sam_options is not None

    detections, boxes, logits, phrases = default_groundingdino(
        image=image,
        caption=sam_options.dino_prompt,
        box_threshold=sam_options.dino_box_threshold,
        text_threshold=sam_options.dino_text_threshold
    )
    # detection_boxes = detections.xyxy
    # # use full image if no box has been found
    # detection_boxes = np.array([[0, 0, image.shape[1], image.shape[0]]]) if len(detection_boxes) == 0 else detection_boxes
    #
    #
    # for idx, box in enumerate(detection_boxes):
    #     box_list = box.tolist()
    #     if box_erode_or_dilate != 0:
    #         box_list[0] -= box_erode_or_dilate
    #         box_list[1] -= box_erode_or_dilate
    #         box_list[2] += box_erode_or_dilate
    #         box_list[3] += box_erode_or_dilate
    #     extras['sam_prompt'] += [{"type": "rectangle", "data": box_list}]
    #
    # if debug_dino:
    #     from PIL import ImageDraw, Image
    #     debug_dino_image = Image.new("RGB", (image.shape[1], image.shape[0]), color="black")
    #     draw = ImageDraw.Draw(debug_dino_image)
    #     for box in extras['sam_prompt']:
    #         draw.rectangle(box['data'], fill="white")
    #     return np.array(debug_dino_image)

    # TODO add support for box_erode_or_dilate again

    H, W = image.shape[0], image.shape[1]
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

    # TODO add model patcher for model logic and device management
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_checkpoint = modules.config.download_sam_model(sam_options.model_type)
    sam = sam_model_registry[sam_options.model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    sam_predictor = SamPredictor(sam)
    final_mask_tensor = torch.zeros((image.shape[0], image.shape[1]))

    if boxes.size(0) > 0:
        sam_predictor.set_image(image)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        masks = optimize_masks(masks)

        num_obj = min(len(logits), sam_options.max_num_boxes)
        for obj_ind in range(num_obj):
            mask_tensor = masks[obj_ind][0]
            final_mask_tensor += mask_tensor

    final_mask_tensor = (final_mask_tensor > 0).to('cpu').numpy()
    mask_image = np.dstack((final_mask_tensor, final_mask_tensor, final_mask_tensor)) * 255
    mask_image = np.array(mask_image, dtype=np.uint8)
    return mask_image
