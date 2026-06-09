# وثيقة ملف: DAT.py
## 1. الوظيفة الأساسية للملف
يحتوي على هيك DAT (Dual Aggregation Transformer) لتحسين جودة الصور.

## 2. الدوال والكلاسات (Functions & Classes)
- DAT: كلاس رئيسي لهيكل DAT
- DATBlock: كلاس ل:block DAT
- ChannelAttention: كلاس لانتباه القنوات
- SpatialAttention: كلاس لانتباه المكاني

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/__init__.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - ldm_patched/pfn/architecture/block.py