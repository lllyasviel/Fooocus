# وثيقة ملف: attention.py
## 1. الوظيفة الأساسية للملف
يحتوي على طبقات الانتباه المختلفة المستخدمة في نماذج التوليد مثل SpatialTransformer وCrossAttention.

## 2. الدوال والكلاسات (Functions & Classes)
- exists(): دالة للتحقق مما إذا كان المدخل موجودًا
- uniq(): دالة لإرجاع العناصر الفريدة من المصفوفة
- default(): دالة لإرجاع القيمة الافتراضية إذا كان المدخل غير موجود
- max_neg_value(): دالة لإرجاع أكبر قيمة سالبة ممكنة للنوع
- init_(): دالة لتهيئة tensor بقيم عشوائية
- zero_module(): دالة لتهيئة وحدة بالقيم الصفرية
- optimized_attention(): دالة للانتباه المحسّن مع دعم xformers
- SpatialTransformer: كلاس للانتباه المكاني
- CrossAttention: كلاس للانتباه المتقاطع
- BasicTransformerBlock: كلاس لل blokc المحول الأساسي
- SpatialTransformer: كلاس للانتباه المكاني

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/ldm/modules/diffusionmodules/openaimodel.py
  - ldm_patched/ldm/modules/diffusionmodules/model.md
  - ldm_patched/controlnet/cldm.py
  - ldm_patched/contrib/external_sag.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - math
  - torch
  - torch.nn.functional
  - torch.nn
  - einops
  - ldm_patched/ldm/modules/diffusionmodules/util.py
  - ldm_patched/ldm/modules/sub_quadratic_attention.py
  - ldm_patched/modules/model_management.py
  - ldm_patched/modules/args_parser.py
  - ldm_patched/modules/ops.py