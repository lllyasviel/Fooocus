# وثيقة ملف: sdxl_clip.md
## 1. الوظيفة الأساسية للملف
مشفر CLIP المزدوج لـ Stable Diffusion XL مع دعم CLIP-G و OpenCLIP.

## 2. الدوال والكلاسات (Functions & Classes)
- `SDXLClipModel()`: نموذج CLIP المزدوج لـ SDXL
- `encode_text()`: ترميز النص بكل النماذج
- `get_embeddings()`: الحصول على التضمينات
- `merge_embeddings()`: دمج التضمينات من النماذج المختلفة
- `pad_tokens()`: تعبئة الرموز

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd.py`, `sample.py`, `conds.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `clip_model.py`, `sd1_clip.py`, `ops.py`
