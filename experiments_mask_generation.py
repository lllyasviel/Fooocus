# https://github.com/sail-sg/EditAnything/blob/main/sam2groundingdino_edit.py

import numpy as np
from PIL import Image

from extras.inpaint_mask import SAMOptions, generate_mask_from_image

original_image = Image.open('cat.webp')
image = np.array(original_image, dtype=np.uint8)

sam_options = SAMOptions(
    dino_prompt='eye',
    dino_box_threshold=0.3,
    dino_text_threshold=0.25,
    dino_erode_or_dilate=0,
    dino_debug=False,
    max_detections=2,
    model_type='vit_b'
)

mask_image, _, _, _ = generate_mask_from_image(image, sam_options=sam_options)

merged_masks_img = Image.fromarray(mask_image)
merged_masks_img.show()
