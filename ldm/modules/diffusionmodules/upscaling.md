# وثيقة ملف: upscaling.py
## 1. الوظيفة الأساسية للملف
يحتوي على طبقات Upsampling لـ UNet في نماذج الـ diffusion.

## 2. الدوال والكلاسات (Functions & Classes)
- Upsample: كلاس لـ upsampling في نموذج الـ diffusion مع دعم resize nearest و interpolation

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/ldm/modules/diffusionmodules/openaimodel.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - torch.nn.functional