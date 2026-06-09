# وثيقة ملف: external_post_processing.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في المعالجة اللاحقة للصور مثل الدمج والضبط اللوني.

## 2. الدوال والكلاسات (Functions & Classes)
- Blend: كلاس لدمج صورتين باستخدام أنماط مختلفة مثل normal وmultiply وscreen
- ImageBlend: كلاس لدمج صورتين بمعامل محدد
- ImageScaleToTotalPixels: كلاس لتكبير الصورة إلى عدد البكسل الكلي
- ImageVaeEncode: كلاس لترميز الصورة باستخدام VAE
- ImageVaeDecode: كلاس لفك ترميز الصورة باستخدام VAE
- LatentToImage: كلاس لتحويل latent إلى صورة
- ImageToLatent: كلاس لتحويل صورة إلى latent

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - numpy
  - torch
  - torch.nn.functional
  - PIL
  - math
  - ldm_patched/modules/utils.py