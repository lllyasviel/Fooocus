import cv2
import numpy as np


def canny_k(x, k=0.5):
    import cv2
    H, W, C = x.shape
    Hs, Ws = int(H * k), int(W * k)
    small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
    return cv2.Canny(small, 100, 200).astype(np.float32) / 255.0


def canny_pyramid(x):
    # For some reasons, SAI's Control-lora Canny seems to be trained on canny maps with non-standard resolutions.
    # Then we use pyramid to use all resolutions to avoid missing any structure in specific resolutions.

    ks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cs = [canny_k(x, k) for k in ks]
    cur = None

    for c in cs:
        if cur is None:
            cur = c
        else:
            H, W = c.shape
            cur = cv2.resize(cur, (W, H), interpolation=cv2.INTER_LINEAR)
            cur = cur * 0.75 + c * 0.25

    cur *= 400.0

    return cur.clip(0, 255).astype(np.uint8)


def cpds(x):
    import cv2
    # cv2.decolor is not "decolor", it is Cewu Lu's method
    # See http://www.cse.cuhk.edu.hk/leojia/projects/color2gray/index.html
    # See https://docs.opencv.org/3.0-beta/modules/photo/doc/decolor.html

    raw = cv2.GaussianBlur(x, (0, 0), 0.8)
    density, boost = cv2.decolor(raw)

    raw = raw.astype(np.float32)
    density = density.astype(np.float32)
    boost = boost.astype(np.float32)

    offset = np.sum((raw - boost) ** 2.0, axis=2) ** 0.5

    result = density + offset

    v_min = np.percentile(result, 4)
    v_max = np.percentile(result, 96)

    result -= v_min
    result /= v_max - v_min

    result *= 255.0

    return result.clip(0, 255).astype(np.uint8)
