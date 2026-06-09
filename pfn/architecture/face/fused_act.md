# وثيقة ملف: fused_act.py
## 1. الوظيفة الأساسية للملف
يحتوي على تطبيق Fused Activation لتحسين كفاءة العمليات Activation.

## 2. الدوال والكلاسات (Functions & Classes)
- FReLU: كلاس لـ Funnel ReLU
- FusedLeakyReLU: كلاس لـ Fused LeakyReLU
- fused_leaky_relu(): دالة مساعدة لتطبيق Fused LeakyReLU

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/face/gfpganv1_arch.py
  - ldm_patched/pfn/architecture/face/gfpganv1_clean_arch.py
  - ldm_patched/pfn/architecture/face/stylegan2_arch.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn