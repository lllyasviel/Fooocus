# وثيقة ملف: patch_precision.md
## 1. الوظيفة الأساسية للملف
تطبيق تلبيسات الدقة لتوفير التوافق مع Kohya. يعدل دوال time embedding و register schedule لتقليل الفروقات بين التدريب والاستدلال.

## 2. الدوال والكلاسات (Functions & Classes)
- **patched_timestep_embedding()**: دالة time embedding معدّلة تتوافق مع تنفيذ Kohya.
- **patched_register_schedule()**: دالة جدول التسجيل معدّلة لحساب sigmas وalphas_cumprod بشكل متوافق.
- **patch_all_precision()**: تطبيق جميع تلبيسات الدقة على وحدات ldm_patched.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/patch.py
- الملفات التي يستدعيها هذا الملف لتشغيله: ldm_patched/ldm/modules/diffusionmodules/openaimodel.py, ldm_patched/modules/model_sampling.py