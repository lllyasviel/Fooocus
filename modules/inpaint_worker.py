import numpy as np

from PIL import Image, ImageFilter
from modules.util import resample_image


current_task = None


def morphological_soft_open(x):
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.MaxFilter(27))
    x = x.filter(ImageFilter.BoxBlur(13))
    x = np.array(x)
    return x


def threshold_0_255(x):
    y = np.zeros_like(x)
    y[x > 127] = 255
    return y


def morphological_hard_open(x):
    y = threshold_0_255(x)
    z = morphological_soft_open(x)
    z[y > 127] = 255
    return z


def imsave(x, path):
    x = Image.fromarray(x)
    x.save(path)


def mask_to_float(x):
    return x.astype(np.float32) / 255.0


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


def solve_abcd(x, a, b, c, d, k):
    H, W = x.shape[:2]
    min_area = H * W * k
    while area_abcd(a, b, c, d) < min_area:
        if (b - a) < (d - c):
            a -= 1
            b += 1
        else:
            c -= 1
            d += 1
        a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


class InpaintWorker:
    def __init__(self, image, mask):
        # mask processing
        self.image_raw = image
        self.mask_raw_user_input = mask
        self.mask_raw_soft = morphological_hard_open(mask)
        self.mask_raw_fg = (self.mask_raw_soft == 255).astype(np.uint8) * 255
        self.mask_raw_bg = (self.mask_raw_soft == 0).astype(np.uint8) * 255
        self.mask_raw_trim = 255 - np.maximum(self.mask_raw_fg, self.mask_raw_bg)
        self.mask_raw_error = (self.mask_raw_user_input > self.mask_raw_fg).astype(np.uint8) * 255

        # log all images
        # imsave(self.mask_raw_user_input, 'mask_raw_user_input.png')
        # imsave(self.mask_raw_soft, 'mask_raw_soft.png')
        # imsave(self.mask_raw_fg, 'mask_raw_fg.png')
        # imsave(self.mask_raw_bg, 'mask_raw_bg.png')
        # imsave(self.mask_raw_trim, 'mask_raw_trim.png')
        # imsave(self.mask_raw_error, 'mask_raw_error.png')

        # mask to float
        # self.mask_raw_user_input = mask_to_float(self.mask_raw_user_input)
        self.mask_raw_soft = mask_to_float(self.mask_raw_soft)
        self.mask_raw_fg = mask_to_float(self.mask_raw_fg)
        self.mask_raw_bg = mask_to_float(self.mask_raw_bg)
        self.mask_raw_trim = mask_to_float(self.mask_raw_trim)
        # self.mask_raw_error = mask_to_float(self.mask_raw_error)

        # compute abcd
        a, b, c, d = compute_initial_abcd(self.mask_raw_bg < 0.5)
        a, b, c, d = solve_abcd(self.mask_raw_bg, a, b, c, d, k=0.4)

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
        self.mask_ready_soft = resample_image(self.mask_interested_soft, W, H)
        return

    def visualize_mask_processing(self):
        result = self.image_raw // 4
        a, b, c, d = self.interested_area
        result[a:b, c:d] += 64
        result[self.mask_raw_trim > 0.5] += 64
        result[self.mask_raw_fg > 0.5] += 128
        mask_vis = (np.ones_like(self.image_raw).astype(np.float32) * self.mask_raw_soft[:, :, None] * 255).astype(np.uint8)
        return [result, mask_vis, self.image_ready, (self.mask_ready_soft * 255).astype(np.uint8)]

