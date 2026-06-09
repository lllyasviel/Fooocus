# وثيقة ملف: HAT.py
## 1. الوظيفة الأساسية للملف
يحتوي على هيكل HAT (Hybrid Attention Transformer) لتحسين جودة الصور.

## 2. الدوال والكلاسات (Functions & Classes)
- HAT: كلاس رئيسي لهيكل HAT
- HATBlock: كلاس ل:block HAT
- ChannelAttention: كلاس لانتباه القنوات
- SpatialAttention: كلاس لانتباه المكاني

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/__init__.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - ldm_patched/pfn/architecture/block.py