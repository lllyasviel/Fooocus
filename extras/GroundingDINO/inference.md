---
### 📄 مسار الملف الافتراضي: docs/extras/GroundingDINO/inference.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
غلاف GroundingDINO للكشف عن الكائنات الموجّه بالنص - يكشف كائنات في الصور بناءً على تعليقات نصية.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **GroundingDinoModel(Model)**: يُغلّف GroundingDINO مع تحميل كسول.
  - `predict_with_caption(self, image, caption, box_threshold, text_threshold)`: كشف الكائنات.
- **predict(model, image, caption, ...)**: الاستدلال الأساسي: tokenize → forward → تصفية → استخراج عبارات.
- **default_groundingdino**: نموذج singleton مُريح.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: torch, groundingdino, supervision, ldm_patched/modules/model_management.py
- **يُستخدم بواسطة**: extras/inpaint_mask.py
- **الأثر البرمجي**: يُمكّن من الكشف عن الكائنات بالنص
