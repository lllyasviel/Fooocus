import cv2


def show_preview(flag, img, title=None):
    if title is None:
        title = flag
    cv2.imshow(flag, img)
    cv2.setWindowTitle(flag, title)
    cv2.waitKey(1)


def close_all_preview():
    cv2.destroyAllWindows()
