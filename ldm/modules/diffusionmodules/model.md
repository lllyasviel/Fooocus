# وثيقة ملف: model.py
## 1. الوظيفة الأساسية للملف
يحتوي على النماذج الأساسية للـ diffusion modules مثل UNet وAttentionBlock وResnetBlock.

## 2. الدوال والكلاسات (Functions & Classes)
- AttentionBlock: كلاس ل:block الانتباه في نموذج الـ diffusion
- ResnetBlock: كلاس ل:block الباقي في نموذج الـ diffusion
- Upsample: كلاس لـ upsampling في نموذج الـ diffusion
- Downsample: كلاس لـ downsampling في نموذج الـ diffusion
- UNetModel: كلاس UNet الرئيسي لنماذج الـ diffusion
- Encoder: كلاس للمُرمِّز في نموذج الـ diffusion
- Decoder: كلاس للفاكم في نموذج الـ diffusion

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/modules/sd.py
  - ldm_patched/ldm/modules/diffusionmodules/openaimodel.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - torch.nn.functional
  - math
  - ldm_patched/ldm/modules/attention.py
  - ldm_patched/ldm/modules/diffusionmodules/util.py