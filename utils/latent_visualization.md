# وثيقة ملف: latent_visualization.md
## 1. الوظيفة الأساسية للملف
يحتوي على دوال لتصور الـ latents وتحويلها إلى صور قابلة للعرض.

## 2. الدوال والكلاسات (Functions & Classes)
- latent_to_rgb(): دالة لتحويل latent إلى صورة RGB
- latent_to_image(): دالة لتحويل latent إلى صورة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/contrib/external.py
  - ldm_patched/contrib/external_custom_sampler.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - numpy