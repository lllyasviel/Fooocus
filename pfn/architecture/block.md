# وثيقة ملف: block.py
## 1. الوظيفة الأساسية للملف
يحتوي على الكتل الأساسية المشتركة لبناء هياكل النماذج المختلفة.

## 2. الدوال والكلاسات (Functions & Classes)
- ResidualBlock: كلاس ل:block الباقي
- ConvBlock: كلاس ل:block الالتوافي
- Upsample: كلاس لـ upsampling
- Downsample: كلاس لـ downsampling
- Normalize: كلاس للتطبيع

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/DAT.md
  - ldm_patched/pfn/architecture/HAT.md
  - ldm_patched/pfn/architecture/RRDB.md
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn