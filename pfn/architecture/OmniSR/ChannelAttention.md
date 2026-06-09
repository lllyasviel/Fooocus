# وثيقة ملف: ChannelAttention.py
## 1. الوظيفة الأساسية للملف
يحتوي على كلاس ChannelAttention لانتباه القنوات في هيكل OmniSR.

## 2. الدوال والكلاسات (Functions & Classes)
- ChannelAttention: كلاس لانتباه القنوات مع SE Block

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/OmniSR/OmniSR.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn