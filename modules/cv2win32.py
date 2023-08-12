import threading
import cv2
import os


buffer = []


def worker():
    global buffer
    try:
        while True:
            cv2.waitKey(50)
            if len(buffer) > 0:
                task = buffer.pop(0)
                if task is None:
                    cv2.destroyAllWindows()
                else:
                    flag, img, title = task
                    cv2.imshow(flag, img)
                    cv2.setWindowTitle(flag, title)
                    cv2.setWindowProperty(flag, cv2.WND_PROP_TOPMOST, 1)
    except Exception as e:
        print('Failed to open preview window. You are not using a local device with GUI support.')
        print(e)
    pass


def write_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[..., ::-1].copy())


def show_preview(flag, img, title='preview'):
    buffer.append((flag, img[..., ::-1].copy(), title))


def close_all_preview():
    buffer.append(None)


threading.Thread(target=worker, daemon=True).start()
