# وثيقة ملف: model_sampling.md
## 1. الوظيفة الأساسية للملف
تعريف جدولة الضوضاء (Noise Schedule) وأنواع التنبؤ المستخدمة في عملية الانتشار.

## 2. الدوال والكلاسات (Functions & Classes)
- `ModelSampling()`: كلاس تنسيق النموذج
- `get_sigmas()`: الحصول على قيم Sigma
- `calculate_sigmas()`: حساب قيم Sigma
- `noise_schedule()`: جدولة الضوضاء
- `prediction_type()`: نوع التنبؤ (epsilon, v-prediction, etc)
- `get_denoise()`: الحصول على عملية إزالة الضوضاء

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `samplers.py`, `sample.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_base.py`, `utils.py`
