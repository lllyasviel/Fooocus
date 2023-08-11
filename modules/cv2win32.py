import cv2


def show_preview(flag, img, title=None):
    if title is None:
        title = flag
    cv2.imshow(flag, img[:, :, ::-1].copy())
    cv2.setWindowTitle(flag, title)
    cv2.setWindowProperty(flag, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)


def close_all_preview():
    cv2.destroyAllWindows()
