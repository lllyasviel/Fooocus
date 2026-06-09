# وثيقة ملف: external_video_model.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في نماذج الفيديو مثل SVD (Stable Video Diffusion).

## 2. الدوال والكلاسات (Functions & Classes)
- ImageOnlyCheckpointLoader: كلاس لتحميل نقاط التحقق الخاصة بنماذج الفيديو
- SVD_img2vid_Conditioning: كلاس لتحضير شروط SVD لتوليد الفيديو من الصور
- SV3D_img2vid_Conditioning: كلاس لتحضير شروط SV3D

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/contrib/external.py
  - torch
  - ldm_patched/modules/utils.py
  - ldm_patched/modules/sd.py
  - ldm_patched/utils/path_utils.py
  - ldm_patched/contrib/external_model_merging.py