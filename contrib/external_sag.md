# وثيقة ملف: external_sag.py
## 1. الوظيفة الأساسية للملف
يُنفذ تقنية Self-Attention Guidance لتحسين جودة التوليد عن طريق تعديل انتباه الذات.

## 2. الدوال والكلاسات (Functions & Classes)
- attention_basic_with_sim(): دالة للانتباه الأساسية مع إرجاع درجات الانتباه
- SAG: كلاس لتطبيق تقنية Self-Attention Guidance على النموذج مع إعداد scale

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn.functional
  - math
  - einops
  - ldm_patched/ldm/modules/attention.py
  - ldm_patched/modules/samplers.py