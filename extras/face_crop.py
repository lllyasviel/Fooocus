import cv2
import numpy as np
import modules.config


faceRestoreHelper = None


def align_warp_face(self, landmark, border_mode='constant'):
    affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
    self.affine_matrices.append(affine_matrix)
    if border_mode == 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode == 'reflect101':
        border_mode = cv2.BORDER_REFLECT101
    elif border_mode == 'reflect':
        border_mode = cv2.BORDER_REFLECT
    input_img = self.input_img
    cropped_face = cv2.warpAffine(input_img, affine_matrix, self.face_size,
                                  borderMode=border_mode, borderValue=(135, 133, 132))
    return cropped_face


def crop_image(img_rgb):
    global faceRestoreHelper
    
    if faceRestoreHelper is None:
        from extras.facexlib.utils.face_restoration_helper import FaceRestoreHelper
        faceRestoreHelper = FaceRestoreHelper(
            upscale_factor=1,
            model_rootpath=modules.config.path_controlnet,
            device='cpu'  # use cpu is safer since we are out of memory management
        )

    faceRestoreHelper.clean_all()
    faceRestoreHelper.read_image(np.ascontiguousarray(img_rgb[:, :, ::-1].copy()))
    faceRestoreHelper.get_face_landmarks_5()

    landmarks = faceRestoreHelper.all_landmarks_5
    # landmarks are already sorted with confidence.

    if len(landmarks) == 0:
        print('No face detected')
        return img_rgb
    else:
        print(f'Detected {len(landmarks)} faces')

    result = align_warp_face(faceRestoreHelper, landmarks[0])

    return np.ascontiguousarray(result[:, :, ::-1].copy())
