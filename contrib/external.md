# وثيقة ملف: external.py
## 1. الوظيفة الأساسية للملف
يحتوي على العقد الأساسية للنظام بما في ذلك عقدة CLIPTextEncode وLoadImage وSaveImage وPreviewImage وغيرها من العقد الأساسية المستخدمة في تدفق العمل.

## 2. الدوال والكلاسات (Functions & Classes)
- before_node_execution(): دالة تُنفَّذ قبل تنفيذ كل عقدة للتحقق من عدم انقطاع المعالجة
- interrupt_processing(): دالة لإيقاف المعالجة الحالية
- CLIPTextEncode: كلاس لترميز النص باستخدام نموذج CLIP
- LoadImage: كلاس لتحميل الصور من المجلد المحدد
- SaveImage: كلاس لحفظ الصور المولدة
- PreviewImage: كلاس لمعاينة الصور المولدة
- ImageBatch: كلاس لإنشاء مجموعة صور
- LatentBatch: كلاس لإنشاء مجموعة latents
- CheckpointLoaderSimple: كلاس لتحميل نقاط التحقق (checkpoints)

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/contrib/external_images.py
  - ldm_patched/contrib/external_mask.py
  - ldm_patched/contrib/external_video_model.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/modules/diffusers_load.py
  - ldm_patched/modules/samplers.py
  - ldm_patched/modules/sample.py
  - ldm_patched/modules/sd.py
  - ldm_patched/modules/utils.py
  - ldm_patched/modules/controlnet.py
  - ldm_patched/modules/clip_vision.py
  - ldm_patched/modules/model_management.py
  - ldm_patched/utils/path_utils.py
  - ldm_patched/utils/latent_visualization.py