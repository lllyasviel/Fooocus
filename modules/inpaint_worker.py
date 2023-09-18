import numpy as np

from PIL import Image, ImageFilter


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


class InpaintWorker:
    def __init__(self, image, mask):

        # mask processing
        imsave(mask, 'raw.png')
        mask = morphological_hard_open(mask)
        imsave(mask, 'after.png')


        # Fooocus inpaint logic
        # 1. ensure that diffusion area cover all masks.
        # 2. ensure that diffusion area cover at lease 1/4 part of images.
        # 3. ensure that diffusion area has at least 1k resolution (by resampling).

        self.raw_image = image
        self.raw_mask = mask
        raise NotImplemented


image = np.load('D:/tmps/image.npy')
mask = np.load('D:/tmps/mask.npy')
InpaintWorker(image=image, mask=mask)
