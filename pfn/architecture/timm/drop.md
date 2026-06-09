# وثيقة ملف: drop.py
## 1. الوظيفة الأساسية للملف
يحتوي على طبقات الإسقاط (Drop) مثل Dropout وDropPath لتحسين التعميم.

## 2. الدوال والكلاسات (Functions & Classes)
- DropPath: كلاس لـ DropPath لتحسين التعميم
- DropPath2d: كلاس لـ DropPath2d
- drop_path(): دالة لتطبيق DropPath

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/DAT.py
  - ldm_patched/pfn/architecture/HAT.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn