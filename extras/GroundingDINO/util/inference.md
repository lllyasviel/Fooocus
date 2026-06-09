---
### 📄 docs/extras/GroundingDINO/util/inference.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
غلاف عملي (wrapper) لنموذج GroundingDINO للكشف عن الكائنات الموجه بالنص. يُنشئ نموذج GroundingDINO مع تحميل كسول (lazy loading)، يُشغّل الاستدلال: tokenize النص → forward pass → تصفية الصناديق حسب عتبات الثقة والنص → استخراج العبارات (phrases) المقابلة. يُستخدم بشكل أساسي من inpaint_mask.py للكشف عن الكائنات قبل تقطيع SAM.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class GroundingDinoModel(Model)**: يغلّف GroundingDINO مع تحميل كسول.
  - `predict_with_caption(self, image, caption, box_threshold, text_threshold)`: كشف الكائنات مع caption نصي.
- **predict(model, image, caption, box_threshold, text_threshold, device)**: دالة الاستدلال الأساسية. tokenize النص (إضافة '.' لكل كلمة)، تشغيل forward، تصفية النتائج، إرجاع (boxes, logits, phrases).
- **default_groundingdino(model_dir)**: نموذج singleton (مُخزَّن مؤقتاً في متغير عام _default) للاستخدام المباشر. يُحمّل النموذج من المسار المُعدّ في config.py.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, groundingdino (GroundingDINO, load_model, predict), supervision, ldm_patched/modules/model_management.py, modules/config.py
- **يُستدعى بواسطة**: extras/inpaint_mask.py (generate_mask_from_image)
- **الأثر البرمجي**: أساسي لتوليد أقنعة Inpainting عبر الكشف النصي
