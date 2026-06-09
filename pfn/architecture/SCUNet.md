# وثيقة ملف: SCUNet.py
## 1. الوظيفة الأساسية للملف
يحتوي على هيكل SCUNet لتنقية الصور وتحسين جودتها.

## 2. الدوال والكلاسات (Functions & Classes)
- SCUNet: كلاس رئيسي لهيكل SCUNet
- SCUNetBlock: كلاس ل:block SCUNet
- ChannelAttention: كلاس لانتباه القنوات
- SpatialAttention: كلاس لانتباه المكاني

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/__init__.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn