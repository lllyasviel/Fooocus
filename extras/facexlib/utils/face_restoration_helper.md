---
### 📄 docs/extras/facexlib/utils/face_restoration_helper.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
فئة FaceRestoreHelper التي تُدير عملية استعادة الوجه بالكامل: قراءة الصورة، كشف الوجوه (بـ RetinaFace)، استخراج نقاط المعالم الخمسية، محاذاة الوجه (Affine)، دمج الوجوه المستعادة في الصورة الأصلية. تُستخدم من face_crop.py لقص الوجه الأكبر.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **get_largest_face(det_faces, h, w)**: إرجاع مؤشر أكبر وجه (أعلى مساحة صندوق) من قائمة الاكتشافات.
- **get_center_face(det_faces, h, w)**: إرجاع مؤشر الوجه الأقرب لمركز الصورة.
- **class FaceRestoreHelper**: فئة إدارة استعادة الوجه.
  - `__init__(self, upscale_factor, face_size, crop_ratio, det_model, save_ext, use_parse, device)`: تهيئة الإعدادات والمعاملات.
  - `set_image_and_size(self, img, img_is_GPU)`: تعيين الصورة وأبعادها الأصلية.
  - `read_image(self, img_path)`: قراءة الصورة من مسار ملف.
  - `get_face_landmarks_5(self)**: كشف الوجوه، استخراج landmarks لأول 4 وجوه.
  - `align_warp_face(self, save_cropped_path=None, skip_align=False)**: محاذاة الوجوه عبر التحويل الأفيني وقصها.
  - `get_inverse_affine(self, save_inverse_affine_path=None)**: حساب التحويل الأفيني العكسي.
  - `add_restored_face(self, face, index)**: إضافة وجه مستعاد.
  - `paste_faces_to_input_image(self)**: لصق الوجوه المستعادة في الصورة الأصلية مع مزج (blending).
  - `clean_all(self)**: تنظيف جميع البيانات المؤقتة.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/facexlib/detection/__init__.py (init_detection_model), extras/facexlib/parsing/__init__.py, extras/facexlib/utils/misc.py, cv2, numpy, torch
- **يُستدعى بواسطة**: extras/face_crop.py (crop_image), extras/facexlib/utils/__init__.py
- **الأثر البرمجي**: أساسي لعمل face_crop.py واستخراج الوجوه للـ Image Prompt
