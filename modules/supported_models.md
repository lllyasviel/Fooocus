# وثيقة ملف: supported_models.md
## 1. الوظيفة الأساسية للملف
تعريف إعدادات وتكوينات النماذج المدعومة في النظام.

## 2. الدوال والكلاسات (Functions & Classes)
- `ModelConfig()`: كلاس إعدادات النموذج
- `get_supported_models()`: الحصول على النماذج المدعومة
- `register_model()`: تسجيل نموذج جديد
- `validate_config()`: التحقق من صحة الإعدادات
- `get_config_for_model()`: الحصول على إعدادات نموذج محدد
- `list_models()`: قائمة النماذج المتاحة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd.py`, `model_detection.py`, `model_base.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `supported_models_base.py`
