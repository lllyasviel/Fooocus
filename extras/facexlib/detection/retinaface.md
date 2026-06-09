---
### 📄 docs/extras/facexlib/detection/retinaface.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج RetinaFace لكشف الوجوه مع نقاط المعالم الخمسية. يُنفّذ شبكة CNN هرمية (Feature Pyramid) مع وحدات SSH (Single Stage Headless) للكشف على مقاييس متعددة. يكتشف الوجوه ويحسب صناديق الإحاطة (bboxes) ونقاط المعالم والثقة.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **generate_config(model_name)**: إرجاع إعدادات الشبكة لكل نموذج: mobile0.25, ResNet50, ResNet101.
- **class RetinaFace(nn.Module)**: النموذج الرئيسي لكشف الوجوه.
  - `forward(self, x)`: التشغيل الأمامي عبر FPN + SSH، إرجاع قائمة خرائط ميزات.
  - `detect_faces(self, image, img_shape, threshold, scale, score_threshold)**: معالجة كاملة للكشف: تحجيم، forward، فك تشفير الصناديق مع NMS، إرجاع قائمة (bbox, landmarks, confidence).
  - `align_multi(self, image, face_boxes, output_size, scale)`: قص ومحاذاة وجوه متعددة.
  - `batched_detect_faces(self, images, img_shapes, thresholds, scales, score_threshold)**: كشف دفعة من الصور.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/facexlib/detection/retinaface_net.py, extras/facexlib/detection/retinaface_utils.py, extras/facexlib/detection/align_trans.py
- **يُستدعى بواسطة**: extras/facexlib/detection/__init__.py
- **الأثر البرمجي**: قلب نظام كشف الوجوه في facexlib
