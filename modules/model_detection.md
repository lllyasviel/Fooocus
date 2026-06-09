# وثيقة ملف: model_detection.md
## 1. الوظيفة الأساسية للملف
الكشف التلقائي عن نوع وبنية النموذج المحمل من ملفات الوزن.

## 2. الدوال والكلاسات (Functions & Classes)
- `detect_config()`: كشف إعدادات النموذج
- `detect_architecture()`: كشف بنية النموذج
- `detect_model_type()`: كشف نوع النموذج
- `get_model_config()`: الحصول على إعدادات النموذج
- `probe_checkpoint()`: فحص نقطة التحقق
- `guess_model()`: تخمين نوع النموذج

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd.py`, `model_management.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_base.py`, `utils.py`
