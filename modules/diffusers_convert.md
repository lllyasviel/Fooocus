# وثيقة ملف: diffusers_convert.md
## 1. الوظيفة الأساسية للملف
تحويل نماذج Diffusers إلى صيغة متوافقة مع ldm_patched.

## 2. الدوال والكلاسات (Functions & Classes)
- `convert_diffusers_to_ldm()`: التحويل من Diffusers إلى LDM
- `map_diffusers_keys()`: تعيين مفاتيح Diffusers
- `convert_unet()`: تحويل UNet
- `convert_vae()`: تحويل VAE
- `convert_text_encoder()`: تحويل مشفر النص
- `load_diffusers_model()`: تحميل نموذج Diffusers

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `diffusers_load.py`, `sd.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_detection.py`, `utils.py`
