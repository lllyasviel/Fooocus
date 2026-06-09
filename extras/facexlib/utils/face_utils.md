---
### 📄 docs/extras/facexlib/utils/face_utils.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
أدوات مساعدة عامة لمعالجة الوجوه: إعادة تحجيم الصور مع الحفاظ على النسبة (resize_and_crop)، إعادة التحجيم مع الحشو للحفاظ على النسبة (resize_and_pad)، وتحويل الصور إلى وحدات معالجة (image2tensor).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **resize_and_crop(img, target_size, center_crop)**: إعادة تحجيم (LANCZOS) وقص central crop حسب الهدف.
- **resize_and_pad(img, target_size)**: إعادة تحجيم مع padding أسود للحفاظ على النسبة.
- **image2tensor(img, bgr2rgb, float32)**: تحويل PIL.Image/Squeeze إلى tensor (3×H×W) مع تطبيع.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: cv2, numpy, torch, PIL
- **يُستدعى بواسطة**: extras/facexlib/utils/face_restoration_helper.py
- **الأثر البرمجي**: أدوات مساعدة لمعالجة الصور في استعادة الوجه
