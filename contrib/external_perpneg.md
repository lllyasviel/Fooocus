# وثيقة ملف: external_perpneg.py
## 1. الوظيفة الأساسية للملف
يُنفذ تقنية Perpendicular Negative Guidance لتحسين جودة التوليد عن طريق تعديل CFG.

## 2. الدوال والكلاسات (Functions & Classes)
- PerpNeg: كلاس لتطبيق تقنية Perpendicular Negative Guidance على النموذج مع إعداد neg_scale

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - ldm_patched/modules/model_management.py
  - ldm_patched/modules/sample.py
  - ldm_patched/modules/samplers.py
  - ldm_patched/modules/utils.py