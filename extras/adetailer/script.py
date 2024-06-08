from extras.adetailer.args import ADetailerArgs
from extras.adetailer.common import get_models, PredictOutput
from extras.adetailer.mask import filter_by_ratio, filter_k_largest, sort_bboxes, mask_preprocess
from modules import config

model_mapping = get_models(
    config.path_adetailer,
    huggingface=True,
)


def get_ad_model(name: str):
    if name not in model_mapping:
        msg = f"[-] ADetailer: Model {name!r} not found. Available models: {list(model_mapping.keys())}"
        raise ValueError(msg)
    return model_mapping[name]


def pred_preprocessing(p, pred: PredictOutput, args: ADetailerArgs, inpaint_only_masked=False):
    pred = filter_by_ratio(
        pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
    )
    pred = filter_k_largest(pred, k=args.ad_mask_k_largest)
    pred = sort_bboxes(pred)
    masks = mask_preprocess(
        pred.masks,
        kernel=args.ad_dilate_erode,
        x_offset=args.ad_x_offset,
        y_offset=args.ad_y_offset,
        merge_invert=args.ad_mask_merge_invert,
    )

    #if inpaint_only_masked:
    # image_mask = self.get_image_mask(p)
    # masks = self.inpaint_mask_filter(image_mask, masks)
    return masks


    # def get_image_mask(p) -> Image.Image:
    #     mask = p.image_mask
    #     if getattr(p, "inpainting_mask_invert", False):
    #         mask = ImageChops.invert(mask)
    #     mask = create_binary_mask(mask)
    #
    #     if is_skip_img2img(p):
    #         if hasattr(p, "init_images") and p.init_images:
    #             width, height = p.init_images[0].size
    #         else:
    #             msg = "[-] ADetailer: no init_images."
    #             raise RuntimeError(msg)
    #     else:
    #         width, height = p.width, p.height
    #     return images.resize_image(p.resize_mode, mask, width, height)