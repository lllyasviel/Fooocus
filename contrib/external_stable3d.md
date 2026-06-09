# وثيقة ملف: external_stable3d.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في نماذج 3D مثل StableZero123 لتوليد مشاهد ثلاثية الأبعاد.

## 2. الدوال والكلاسات (Functions & Classes)
- camera_embeddings(): دالة لإنشاء تضمينات الكاميرا من الارتفاع والazimuth
- StableZero123_Conditioning: كلاس لتحضير شروط StableZero123
- SV3D_Conditioning: كلاس لتحضير شروط SV3D

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - ldm_patched/contrib/external.py
  - ldm_patched/modules/utils.py