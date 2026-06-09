# وثيقة ملف: controlnet.md
## 1. الوظيفة الأساسية للملف
دعم ControlNet و T2I-Adapter للتحكم في عملية توليد الصور باستخدام إشارات التحكم.

## 2. الدوال والكلاسات (Functions & Classes)
- `ControlNet()`: كلاس أساسي لـ ControlNet
- `T2IAdapter()`: كلاس T2I-Adapter
- `apply_controlnet()`: تطبيق ControlNet على النموذج
- `load_controlnet()`: تحميل نموذج ControlNet
- `preprocess_controlnet()`: معالجة مسبقة لإشارات التحكم
- `merge_controlnet()`: دمج إشارات ControlNet المتعددة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sample.py`, `samplers.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_patcher.py`, `ops.py`
