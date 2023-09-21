import os.path

import torch
import numpy as np
import modules.default_pipeline as pipeline

from PIL import Image, ImageFilter
from modules.util import resample_image
from modules.path import inpaint_models_path


inpaint_head = None


class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device='cpu'))

    def __call__(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)


current_task = None


def box_blur(x, k):
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.BoxBlur(k))
    return np.array(x)


def max33(x):
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.MaxFilter(3))
    return np.array(x)


def morphological_open(x):
    x_int32 = np.zeros_like(x).astype(np.int32)
    x_int32[x > 127] = 256
    for _ in range(32):
        maxed = max33(x_int32) - 8
        x_int32 = np.maximum(maxed, x_int32)
    return x_int32.clip(0, 255).astype(np.uint8)


def imsave(x, path):
    x = Image.fromarray(x)
    x.save(path)


def regulate_abcd(x, a, b, c, d):
    H, W = x.shape[:2]
    if a < 0:
        a = 0
    if a > H:
        a = H
    if b < 0:
        b = 0
    if b > H:
        b = H
    if c < 0:
        c = 0
    if c > W:
        c = W
    if d < 0:
        d = 0
    if d > W:
        d = W
    return int(a), int(b), int(c), int(d)


def compute_initial_abcd(x):
    indices = np.where(x)
    a = np.min(indices[0]) - 64
    b = np.max(indices[0]) + 65
    c = np.min(indices[1]) - 64
    d = np.max(indices[1]) + 65
    a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


def area_abcd(a, b, c, d):
    return (b - a) * (d - c)


def solve_abcd(x, a, b, c, d, k, outpaint):
    H, W = x.shape[:2]
    if outpaint:
        return 0, H, 0, W
    min_area = H * W * k
    max_area = H * W
    while True:
        if area_abcd(a, b, c, d) > min_area and abs((b - a) - (d - c)) < 16:
            break
        if area_abcd(a, b, c, d) >= max_area:
            break

        add_h = (b - a) < (d - c)
        add_w = not add_h

        if b - a == H:
            add_w = True

        if d - c == W:
            add_h = True

        if add_h:
            a -= 1
            b += 1

        if add_w:
            c -= 1
            d += 1

        a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


def fooocus_fill(image, mask):
    current_image = image.copy()
    raw_image = image.copy()
    area = np.where(mask < 127)
    store = raw_image[area]

    for k, repeats in [(64, 4), (32, 4), (16, 4), (4, 4), (2, 4)]:
        for _ in range(repeats):
            current_image = box_blur(current_image, k)
            current_image[area] = store

    return current_image


class InpaintWorker:
    def __init__(self, image, mask, is_outpaint):
        # mask processing
        self.mask_raw_soft = morphological_open(mask)
        self.mask_raw_fg = (self.mask_raw_soft == 255).astype(np.uint8) * 255
        self.mask_raw_bg = (self.mask_raw_soft == 0).astype(np.uint8) * 255
        self.mask_raw_trim = 255 - np.maximum(self.mask_raw_fg, self.mask_raw_bg)

        # image processing
        self.image_raw = fooocus_fill(image, self.mask_raw_fg)

        # log all images
        # imsave(self.image_raw, 'image_raw.png')
        # imsave(self.mask_raw_soft, 'mask_raw_soft.png')
        # imsave(self.mask_raw_fg, 'mask_raw_fg.png')
        # imsave(self.mask_raw_bg, 'mask_raw_bg.png')
        # imsave(self.mask_raw_trim, 'mask_raw_trim.png')

        # compute abcd
        a, b, c, d = compute_initial_abcd(self.mask_raw_bg < 127)
        a, b, c, d = solve_abcd(self.mask_raw_bg, a, b, c, d, k=0.618, outpaint=is_outpaint)

        # interested area
        self.interested_area = (a, b, c, d)
        self.mask_interested_soft = self.mask_raw_soft[a:b, c:d]
        self.mask_interested_fg = self.mask_raw_fg[a:b, c:d]
        self.mask_interested_bg = self.mask_raw_bg[a:b, c:d]
        self.mask_interested_trim = self.mask_raw_trim[a:b, c:d]
        self.image_interested = self.image_raw[a:b, c:d]

        # resize to make images ready for diffusion
        H, W, C = self.image_interested.shape
        k = (1024.0 ** 2.0 / float(H * W)) ** 0.5
        H = int(np.ceil(float(H) * k / 16.0)) * 16
        W = int(np.ceil(float(W) * k / 16.0)) * 16
        self.image_ready = resample_image(self.image_interested, W, H)
        self.mask_ready = resample_image(self.mask_interested_soft, W, H)

        # ending
        self.latent = None
        self.latent_mask = None
        self.inpaint_head_feature = None
        return

    def load_inpaint_guidance(self, latent, mask, model_path):
        global inpaint_head
        if inpaint_head is None:
            inpaint_head = InpaintHead()
            sd = torch.load(model_path, map_location='cpu')
            inpaint_head.load_state_dict(sd)
        process_latent_in = pipeline.xl_base_patched.unet.model.process_latent_in

        latent = process_latent_in(latent)
        B, C, H, W = latent.shape

        mask = torch.nn.functional.interpolate(mask, size=(H, W), mode="bilinear")
        mask = mask.round()

        feed = torch.cat([mask, latent], dim=1)

        inpaint_head.to(device=feed.device, dtype=feed.dtype)
        self.inpaint_head_feature = inpaint_head(feed)
        return

    def load_latent(self, latent, mask):
        self.latent = latent
        self.latent_mask = mask

    def color_correction(self, img):
        fg = img.astype(np.float32)
        bg = self.image_raw.copy().astype(np.float32)
        w = self.mask_raw_soft[:, :, None].astype(np.float32) / 255.0
        y = fg * w + bg * (1 - w)
        return y.clip(0, 255).astype(np.uint8)

    def post_process(self, img):
        a, b, c, d = self.interested_area
        content = resample_image(img, d - c, b - a)
        result = self.image_raw.copy()
        result[a:b, c:d] = content
        result = self.color_correction(result)
        return result

    def visualize_mask_processing(self):
        result = self.image_raw // 4
        a, b, c, d = self.interested_area
        result[a:b, c:d] += 64
        result[self.mask_raw_trim > 127] += 64
        result[self.mask_raw_fg > 127] += 128
        return [result, self.mask_raw_soft, self.image_ready, self.mask_ready]

