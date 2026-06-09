# وثيقة ملف: external_photomaker.py
## 1. الوظيفة الأساسية للملف
يُنفذ نموذج PhotoMaker لتوليد صور شخصيات مخصصة مع الحفاظ على الهوية.

## 2. الدوال والكلاسات (Functions & Classes)
- MLP: كلاس لشبكة عصبية متعددة الطبقات مع دعم الباقي
- FuseModule: كلاس لدمج ميزات CLIP وصورة الشخصية
- PhotoMakerLoader: كلاس لتحميل نموذج PhotoMaker
- PhotoMakerStyleLoader: كلاس لتحميل أسلوب PhotoMaker
- PhotoMakerConditioning: كلاس لتحضير شروط PhotoMaker

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - ldm_patched/utils/path_utils.py
  - ldm_patched/modules/clip_model.py
  - ldm_patched/modules/clip_vision.py
  - ldm_patched/modules/ops.py