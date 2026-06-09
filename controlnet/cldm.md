# وثيقة ملف: cldm.py
## 1. الوظيفة الأساسية للملف
يُنفذ نموذج ControlNet للتحكم في عملية التوليد باستخدام ميزات صورة الإدخال.

## 2. الدوال والكلاسات (Functions & Classes)
- ControlledUnetModel: كلاس UNet المعدل للعمل مع ControlNet
- ControlNet: كلاس رئيسي لنموذج ControlNet
  - __init__(): دالة التهيئة مع جميع المعلمات المطلوبة
  - forward(): الدالة الأمامية لمعالجة الإدخال وإخراج ميزات التحكم

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/modules/controlnet.py
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn
  - ldm_patched/ldm/modules/diffusionmodules/util.py
  - ldm_patched/ldm/modules/attention.py
  - ldm_patched/ldm/modules/diffusionmodules/openaimodel.py
  - ldm_patched/ldm/util.py
  - ldm_patched/modules/ops.py