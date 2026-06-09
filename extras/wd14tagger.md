---
### 📄 docs/extras/wd14tagger.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
تصنيف الصور الأنمي واستخراج الوسوم (tags) باستخدام WD14 Moat Tagger (ONNX). يُحمّل النموذج بصيغة ONNX مع قائمة وسوم CSV، يُعيد تحجيم الصورة مع padding للحفاظ على النسبة (448×448)، يُشغّل الاستدلال (inference)، يُصفّي الوسوم حسب عتبات الثقة (general=0.35, character=0.85)، ويُرجع وسوماً مفصولة بفواصل.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **default_interrogator(image, threshold, general_threshold, exclude_tags, bypass)**: الدالة الرئيسية:
  - تحميل ONNX InferenceSession و CSV tag list.
  - تحجيم الصورة مع padding (PIL.Image.LANCZOS) إلى 448×448.
  - تطبيع وتشغيل الاستدلال عبر session.run.
  - تصفية الوسوم حسب العتبات (general/character).
  - استبعاد الوسوم المحددة في exclude_tags.
  - إرجاع وسوم مفصولة بفواصل (comma-separated).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: modules/config.py, modules/model_loader.py, onnxruntime (InferenceSession), PIL (Image), numpy, csv
- **يُستدعى بواسطة**: webui.py (trigger_describe للوضع Anime), experiments_interrogate.py
- **الأثر البرمجي**: تعطيله يُلغي وصف الصور الأنمي في واجهة المستخدم
