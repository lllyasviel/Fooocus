# وثيقة ملف: clip_model.md
## 1. الوظيفة الأساسية للملف
تعريف بنى الشبكات العصبية لـ CLIP المستخدمة في ترميز النصوص والصور.

## 2. الدوال والكلاسات (Functions & Classes)
- `CLIPTextModel()`: نموذج نص CLIP
- `CLIPVisionModel()`: نموذج رؤية CLIP
- `CLIPModel()`: النموذج المدمج
- `Transformer()`: محول CLIP
- `ResidualAttentionBlock()`: كلاس انتباه متبقٍ
- `MLP()`: شبكة عصبية متعددة الطبقات

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd1_clip.py`, `sd2_clip.py`, `sdxl_clip.py`, `clip_vision.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `ops.py`, `utils.py`
