import cv2
from extras.interrogate import default_interrogator as default_interrogator_photo
from extras.wd14tagger import default_interrogator as default_interrogator_anime

img = cv2.imread('./test_imgs/red_box.jpg')[:, :, ::-1].copy()
print(default_interrogator_photo(img))
img = cv2.imread('./test_imgs/miku.jpg')[:, :, ::-1].copy()
print(default_interrogator_anime(img))
