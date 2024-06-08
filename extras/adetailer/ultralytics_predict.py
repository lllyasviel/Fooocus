from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from extras.adetailer.common import PredictOutput, create_mask_from_bbox

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    apply_classes(model, model_path, classes)
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)
    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes or "-world" not in Path(model_path).stem:
        return
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if parsed:
        model.set_classes(parsed)


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


