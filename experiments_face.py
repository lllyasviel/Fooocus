import cv2
import extras.face_crop as cropper


img = cv2.imread('lena.png')
result = cropper.crop_image(img)
cv2.imwrite('lena_result.png', result)
