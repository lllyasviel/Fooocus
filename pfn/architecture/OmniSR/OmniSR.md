# وثيقة ملف: OmniSR.py
## 1. الوظيفة الأساسية للملف
يحتوي على هيكل OmniSR الرئيسي لتحسين دقة الصور.

## 2. الدوال والكلاسات (Functions & Classes)
- OmniSR: كلاس رئيسي لهيكل OmniSR
- OmniSRBlock: كلاس ل:block OmniSR
- WindowAttention: كلاس لانتباه النافذة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/__init__.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - ldm_patched/pfn/architecture/OmniSR/ChannelAttention.py
  - ldm_patched/pfn/architecture/OmniSR/esa.py
  - ldm_patched/pfn/architecture/OmniSR/layernorm.py
  - ldm_patched/pfn/architecture/OmniSR/OSA.py
  - ldm_patched/pfn/architecture/OmniSR/OSAG.py
  - ldm_patched/pfn/architecture/OmniSR/pixelshuffle.py