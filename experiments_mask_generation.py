import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything.utils.amg import remove_small_regions

from extras.GroundingDINO.util.inference import default_groundingdino
from extras.adetailer.args import ADetailerArgs
from extras.adetailer.script import get_ad_model
from extras.adetailer.script import pred_preprocessing
from extras.adetailer.ultralytics_predict import ultralytics_predict
from extras.inpaint_mask import run_grounded_sam, generate_mask_from_image

original_image1 = cv2.imread('cat.webp')
original_image = Image.fromarray(original_image1)
device = "cuda" if torch.cuda.is_available() else "cpu"

# predictor = ultralytics_predict
#
# ad_model = get_ad_model('face_yolov8n.pt')
#
# kwargs = {}
# kwargs["device"] = torch.device('cpu')
# kwargs["classes"] = ""
#
# img2 = Image.fromarray(img)
# pred = predictor(ad_model, img2, **kwargs)
#
# if pred.preview is None:
#     print('[ADetailer] nothing detected on image')
#
# args = ADetailerArgs()
#
# masks = pred_preprocessing(img, pred, args)
# merged_masks = np.maximum(*[np.array(mask) for mask in masks])
#
#
# merged_masks_img = Image.fromarray(merged_masks)
# merged_masks_img.show()

sam_prompt = 'eye'
sam_model = 'sam_vit_l_0b3195'
dino_box_threshold = 0.3
dino_text_threshold = 0.25
box_erode_or_dilate = 0

detections, boxes, logits, phrases = default_groundingdino(
    image=np.array(original_image),
    caption=sam_prompt,
    box_threshold=dino_box_threshold,
    text_threshold=dino_text_threshold
)

# for boxes.xyxy
#boxes = run_grounded_sam(img, sam_prompt, box_threshold=dino_box_threshold, text_threshold=dino_text_threshold)
#boxes = np.array([[0, 0, img.shape[1], img.shape[0]]]) if len(boxes) == 0 else boxes

# from PIL import ImageDraw
# draw = ImageDraw.Draw(img)
# for idx, box in enumerate(boxes.xyxy):
#     box_list = box.tolist()
#     if box_erode_or_dilate != 0:
#         box_list[0] -= box_erode_or_dilate
#         box_list[1] -= box_erode_or_dilate
#         box_list[2] += box_erode_or_dilate
#         box_list[3] += box_erode_or_dilate
#     draw.rectangle(box_list, fill=128, outline ="red")
# img.show()

H, W = original_image.size[1], original_image.size[0]
boxes = boxes * torch.Tensor([W, H, W, H])
boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "./models/sam/sam_vit_l_0b3195.pth"
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
num_boxes = 2

sam_predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device))

image_np = np.array(original_image, dtype=np.uint8)

final_m = torch.zeros((image_np.shape[0], image_np.shape[1]))

if boxes.size(0) > 0:
    sam_predictor.set_image(image_np)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # remove small disconnected regions and holes
    fine_masks = []
    for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
        fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
    masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
    masks = torch.from_numpy(masks)

    num_obj = min(len(logits), num_boxes)
    for obj_ind in range(num_obj):
        # box = boxes[obj_ind]

        m = masks[obj_ind][0]
        final_m += m
final_m = (final_m > 0).to('cpu').numpy()
# print(final_m.max(), final_m.min())
mask_image = np.array(np.dstack((final_m, final_m, final_m)) * 255, dtype=np.uint8)

merged_masks_img = Image.fromarray(mask_image)
merged_masks_img.show()
