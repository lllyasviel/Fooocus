---
### 📄 docs/extras/facexlib/detection/align_trans.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
توفير دوال محاذاة الوجوه عبر التحويل الأفيني (Affine Transformation). يستخدم نقاط المعالم الخمسية (eye_left, eye_right, nose, mouth_left, mouth_right) مع نقاط مرجعية ثابتة لحساب مصفوفة التحويل، ثم تطبيقها لقص الوجه بحجم 112×112.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **get_reference_facial_points(output_size)**: إرجاع نقاط الوجه المرجعية لحجم إخراج محدد.
- **get_affine_transform_matrix(cropped_face_size, reference_pts, src_pts)**: حساب مصفوفة التحويل الأفيني (2×3) بين نقاط المصدر والمرجع.
- **face_align(src_face, src_pts, output_size)**: تطبيق التحويل الأفيني على الصورة باستخدام warpAffine.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: math, cv2, numpy
- **يُستدعى بواسطة**: extras/facexlib/detection/retinaface.py (RetinaFace.align_multi)
- **الأثر البرمجي**: أساسي لمحاذاة الوجوه المقصوصة في RetinaFace
