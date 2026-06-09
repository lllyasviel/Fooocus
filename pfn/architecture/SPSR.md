# وثيقة ملف: SPSR.py
## 1. الوظيفة الأساسية للملف
يحتوي على هيكل SPSR (Self-Pruned Super-Resolution) لتحسين دقة الصور.

## 2. الدوال والكلاسات (Functions & Classes)
- SPSR: كلاس رئيسي لهيكل SPSR
- SPSRNet: كلاس الشبكة الكاملة SPSR
- RRDBBlock: كلاس ل:block RRDB مع Self-Pruning

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/__init__.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - ldm_patched/pfn/architecture/RRDB.py