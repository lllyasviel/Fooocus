# وثيقة ملف: external_latent.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في عمليات latent مثل الجمع والطرح والضرب والدمج.

## 2. الدوال والكلاسات (Functions & Classes)
- reshape_latent_to(): دالة لإعادة تشكيل latent للحجم المطلوب
- LatentAdd: كلاس لجمع عينتين latent
- LatentSubtract: كلاس لطرح عينتين latent
- LatentMultiply: كلاس لضرب latent بقيمة
- LatentInterpolate: كلاس لاستيفاء بين عينتين latent
- LatentComposite: كلاس لدمج عينتين latent في مواقع مختلفة
- LatentBatch: كلاس لإنشاء مجموعة من latent
- LatentUpscale: كلاس لتكبير latent باستخدام طرق مختلفة
- LatentUpscaleBy: كلاس لتكبير latent بنسب محددة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/modules/utils.py
  - torch