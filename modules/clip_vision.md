# وثيقة ملف: clip_vision.md
## 1. الوظيفة الأساسية للملف
تحميل واستخدام نموذج CLIP Vision لتضمين الصور واستخراج الميزات البصرية.

## 2. الدوال والكلاسات (Functions & Classes)
- `CLIPVisionModel()`: نموذج CLIP Vision
- `encode_image()`: ترميز الصورة
- `load_clip_vision()`: تحميل نموذج CLIP Vision
- `preprocess_image()`: معالجة مسبقة للصور
- `get_image_features()`: الحصول على ميزات الصورة
- `project_features()`: إسقاط الميزات

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sample.py`, `clip_model.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `ops.py`, `utils.py`
