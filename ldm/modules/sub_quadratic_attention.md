# وثيقة ملف: sub_quadratic_attention.py
## 1. الوظيفة الأساسية للملف
يُنفذ خوارزمية انتباه فعالة من حيث الذاكرة تتجنب الحاجة إلى O(n^2) من الذاكرة.

## 2. الدوال والكلاسات (Functions & Classes)
- dynamic_slice(): دالة للقطع الديناميكي بناءً على الأبعاد
- AttnChunk: كلاس namedtuple لإخراج chunk الانتباه
- SummarizeChunk: بروتوكول لدالة تلخيص chunk
- ComputeQueryChunkAttn: بروتوكول لدالة حساب انتباه chunk الاستعلام
- efficient_dot_product_attention(): دالة للانتباه الفعال من حيث الذاكرة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/ldm/modules/attention.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - functools
  - torch
  - torch.utils.checkpoint
  - math
  - typing
  - ldm_patched/modules/model_management.py