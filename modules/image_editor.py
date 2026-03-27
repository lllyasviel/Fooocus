"""OpenCV-based image adjustment utilities for the gallery editor."""
import cv2
import numpy as np


# ────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────

def resize_for_preview(img_array, max_size=768):
    """プレビュー用にアスペクト比を保ってリサイズ（高速化）。"""
    if img_array is None:
        return None
    h, w = img_array.shape[:2]
    if max(h, w) <= max_size:
        return img_array
    scale = max_size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img_array, (nw, nh), interpolation=cv2.INTER_AREA)


def rotate_image(img_array, degrees):
    """画像を 90 / 180 / 270 度回転（RGB 入出力）。"""
    if img_array is None:
        return None
    degrees = int(degrees) % 360
    if degrees == 90:
        return cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img_array, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_array


def flip_image(img_array, direction='h'):
    """水平('h') または垂直('v') 反転（RGB 入出力）。"""
    if img_array is None:
        return None
    code = 1 if direction == 'h' else 0
    return cv2.flip(img_array, code)


# ────────────────────────────────────────────────
# メイン調整関数
# ────────────────────────────────────────────────

def apply_adjustments(img_array,
                      brightness=0, contrast=1.0,
                      saturation=1.0, hue_shift=0,
                      sharpness=1.0, temperature=0):
    """OpenCV で画像調整を適用する。

    Parameters
    ----------
    img_array   : numpy ndarray (H, W, 3)  RGB
    brightness  : int    -100 … 100   加算
    contrast    : float   0.1 … 3.0   乗算
    saturation  : float   0.0 … 3.0   HSV S 倍率
    hue_shift   : int    -90 … 90    HSV H 加算（度）
    sharpness   : float   0.0 … 3.0   1.0=変化なし / >1=鮮明 / <1=ぼかし
    temperature : int   -100 … 100   負=クール(青) / 正=ウォーム(赤)

    Returns
    -------
    numpy ndarray (H, W, 3) RGB
    """
    if img_array is None:
        return None

    img = np.clip(img_array, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ── 明るさ / コントラスト ──────────────────────
    if brightness != 0 or contrast != 1.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=float(contrast), beta=float(brightness))

    # ── 彩度 / 色相 ────────────────────────────────
    if saturation != 1.0 or hue_shift != 0:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + float(hue_shift)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 色温度 ─────────────────────────────────────
    if temperature != 0:
        b, g, r = cv2.split(bgr.astype(np.float32))
        t = float(temperature)
        if t > 0:          # ウォーム (赤+, 青-)
            r = np.clip(r + t * 0.8, 0, 255)
            b = np.clip(b - t * 0.4, 0, 255)
        else:              # クール (青+, 赤-)
            r = np.clip(r + t * 0.4, 0, 255)
            b = np.clip(b - t * 0.8, 0, 255)
        bgr = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

    # ── シャープネス（アンシャープマスク）─────────
    if sharpness != 1.0:
        blur = cv2.GaussianBlur(bgr, (0, 0), 3.0)
        strength = float(sharpness) - 1.0
        bgr = cv2.addWeighted(bgr, 1.0 + strength, blur, -strength, 0)
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
