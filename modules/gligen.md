# وثيقة ملف: gligen.md
## 1. الوظيفة الأساسية للملف
دعم GLIGEN (Grounded Language-to-Image Generation) للتحكم في توليد الصور باستخدام النصوص المرتبطة بالمواقع.

## 2. الدوال والكلاسات (Functions & Classes)
- `GLIGEN()`: كلاس GLIGEN الرئيسي
- `apply_gligen()`: تطبيق GLIGEN على النموذج
- `load_gligen()`: تحميل نموذج GLIGEN
- `encode_boxes()`: ترميز صناديق التحديد
- `process_gligen()`: معالجة مدخلات GLIGEN
- `set_gligen_parameters()`: تعيين معلمات GLIGEN

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sample.py`, `samplers.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_patcher.py`, `ops.py`
