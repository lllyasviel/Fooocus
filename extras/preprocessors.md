---
### 📄 docs/extras/preprocessors.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
معالجة مسبقة للصور لاستخراج خرائط الحواف (edge maps) كمدخلات لـ ControlNet. يُنفّذ كشف Canny المحسَّن (متعدد القنوات ومتعدد المقاييس الهرمية) وطريقة CPDS (Color-to-Gray) لتحويل الصور الملونة إلى تدرج رمادي مع الحفاظ على تباين الحواف.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **centered_canny(x)**: تطبيق Canny edge detection على قناة واحدة (مستوى رمادي).
- **centered_canny_color(x)**: تطبيق Canny على كل قناة RGB بشكل مستقل، دمج النتائج بالكومة (stack) عبر axis=2.
- **pyramid_canny_color(x, low, high)**: بناء هرم حواف متعدد المقاييس بتشغيل Canny على 9 مقاييس مختلفة (1.0, 0.9, 0.8, ... 0.2) ودمج النتائج بالمتوسط.
- **norm255(x, lo, hi)**: تطبيع مصفوفة float32 إلى النطاق 0-255 باستخدام percentile clipping.
- **canny_pyramid(x)**: دمج pyramid Canny على الصورة الملونة مع التطبيع إلى uint8.
- **cpds(x)**: تحويل الصورة الملونة إلى رمادي باستخدام decolor (OpenCV) مع تحليل الكثافة/التعزيز وتعويض الإزاحة.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: cv2 (OpenCV), numpy
- **يُستدعى بواسطة**: modules/async_worker.py
- **الأثر البرمجي**: تعطيله يُلغي ميزة توليد خرائط الحواف لـ ControlNet
