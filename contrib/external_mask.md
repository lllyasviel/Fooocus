# وثيقة ملف: external_mask.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في معالجة الـ masks مثل الدمج والقص والتغيير الحجم.

## 2. الدوال والكلاسات (Functions & Classes)
- composite(): دالة لدمج source في destination باستخدام mask
- LatentCompositeMasked: كلاس لدمج latent مع mask
- CompositeMasked: كلاس لدمج صورتين مع mask
- MaskToImage: كلاس لتحويل mask إلى صورة
- ImageToMask: كلاس لتحويل صورة إلى mask
- InvertMask: كلاس لعكس mask
- CropMask: كلاس لقص mask بأبعاد محددة
- ResizeMask: كلاس لتغيير حجم mask
- FeatherMask: كلاس لتطبيق تأثير الريش على mask
- ThresholdMask: كلاس لتطبيق عتبة على mask

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - numpy
  - scipy.ndimage
  - torch
  - ldm_patched/modules/utils.py
  - ldm_patched/contrib/external.py (لاستيراد MAX_RESOLUTION)