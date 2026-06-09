# وثيقة ملف: diffusers_load.md
## 1. الوظيفة الأساسية للملف
تحميل نماذج Diffusers من الدليل مع دعم الهيكلية المختلفة.

## 2. الدوال والكلاسات (Functions & Classes)
- `load_diffusers_directory()`: تحميل دليل Diffusers
- `detect_diffusers_config()`: كشف إعدادات Diffusers
- `load_diffusers_pipeline()`: تحميل أنابيب Diffusers
- `get_diffusers_submodels()`: الحصول على النماذج الفرعية
- `validate_diffusers_structure()`: التحقق من هيكل Diffusers

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd.py`, `model_detection.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `diffusers_convert.py`, `model_management.py`
