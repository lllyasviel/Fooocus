---
### 📄 docs/extras/sam/predictor.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
غلاف (wrapper) لـ SAM Predictor (Segment Anything Model) المُحمَّل عبر model_management. يُنشئ نموذج SAM (vit_b/vit_l/vit_h)، يُخزّن تضمينات الصورة مؤقتاً للإسراع، ويُقدّم دالة predict التي تأخذ صناديق إحاطة (boxes) من GroundingDINO وتُعيد أقنعة التقطيع المقابلة. يُستخدم في inpaint_mask.py للتقطيع الدقيق بعد الكشف النصي.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class SamPredictor**: فئة التنبؤ بالقناع.
  - `__init__(self, model)`: تخزين النموذج وتجهيز الجهاز.
  - `set_image(self, image)`: تعيين الصورة (numpy HWC)، تحويلها إلى tensor، حساب تضمين الصورة عبر model.image_encoder، تخزينه للتسريع.
  - `set_torch_image(self, image)`: تعيين الصورة كـ tensor مباشر.
  - `predict(self, boxes)`: التنبؤ بالقناع: تمرير boxes عبر model.prompt_encoder + mask_decoder للحصول على الأقنعة.
  - `predict_torch(self, boxes)`: التنبؤ باستخدام tensors.
  - `get_image_embedding(self)`: إرجاع تضمين الصورة المُخزَّن.
  - `reset_image(self)`: إعادة تعيين الصورة والتضمين.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: segment_anything (sam_model_registry), ldm_patched/modules/model_management.py, ldm_patched/modules/model_patcher.py
- **يُستدعى بواسطة**: extras/inpaint_mask.py (generate_mask_from_image)
- **الأثر البرمجي**: أساسي لتقطيع الكائنات في ميزة توليد الأقنعة
