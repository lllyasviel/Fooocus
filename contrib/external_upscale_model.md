# وثيقة ملف: external_upscale_model.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد لتحميل وتطبيق نماذج التكبير (upscale) مثل RealESRGAN.

## 2. الدوال والكلاسات (Functions & Classes)
- UpscaleModelLoader: كلاس لتحميل نموذج التكبير من ملف
- ImageUpscaleWithModel: كلاس لتطبيق نموذج التكبير على الصور

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - os
  - ldm_patched/pfn/model_loading.py
  - ldm_patched/modules/model_management.py
  - torch
  - ldm_patched/modules/utils.py
  - ldm_patched/utils/path_utils.py