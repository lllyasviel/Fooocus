# وثيقة ملف: patch_clip.md
## 1. الوظيفة الأساسية للملف
تطبيق تلبيسات على مشفر CLIP لتوفير التوافق مع Kohya/A1111. يعدل طريقة تشفير الرموز النصية وتهيئة النموذج لضمان التوافق بين التدريب والاستدلال.

## 2. الدوال والكلاسات (Functions & Classes)
- **patched_encode_token_weights()**: دالة تشفير معدّلة تدعم الأوزان وتتعامل مع الحد الأقصى للطول.
- **patched_SDClipModel__init__()`**: تهيئة معدّلة لنموذج SDClip تستخدم HuggingFace transformers.
- **patch_all_clip()**: تطبيق جميع التلبيسات على وحدات CLIP.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/patch.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/ops.py, ldm_patched/modules/sd1_clip.py, ldm_patched/modules/ops.py, transformers (CLIPTextModel, CLIPVisionModelWithProjection)