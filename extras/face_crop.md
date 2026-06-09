---
### 📄 docs/extras/face_crop.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
قص الوجوه ومحاذاتها من الصور لاكتشاف الوجه الأكبر. يستخدم FaceRestoreHelper من facexlib للكشف (RetinaFace)، استخراج نقاط المعالم الخمسية (5 landmarks)، وتحويل أفيني (affine) لمحاذاة الوجه إلى قالب قياسي (112×112). يُعيد الصورة المقتصّة وأبعادها الأصلية للاستخدام في Image Prompt.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **align_warp_face(landmarks_5, image, output_size)**: تطبيق تحويل أفيني لمحاذاة نقاط المعالم إلى القالب، إرجاع الوجه المقتصّ مع مصفوفة التحويل.
- **crop_image(image)**: نقطة الدخول الرئيسية: يُنشئ FaceRestoreHelper، يكشف الوجوه عبر detect_faces، يستخرج landmarks لأول 4 وجوه، يختار أكبر وجه، يُحوّله أفينياً، يُعيد (cropped_face, orig_size, (orig_w, orig_h)).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: modules/config.py, extras/facexlib/utils/face_restoration_helper.py (FaceRestoreHelper), cv2, numpy
- **يُستدعى بواسطة**: modules/async_worker.py (لاستخراج وجه الصورة المُدخَلة), experiments_face.py
- **الأثر البرمجي**: تعطيله يُعطّل Image Prompt المعتمد على الوجوه
