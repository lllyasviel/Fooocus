# وثيقة ملف: patch.md
## 1. الوظيفة الأساسية للملف
تطبيق التغييرات (monkey-patches) على وحدات ldm_patched لتخصيص سلوك نموذج Stable Diffusion. يعدل طريقة عمل العينات والانتباه والتحكم في التشتت وغير ذلك من المكونات الأساسية.

## 2. الدوال والكلاسات (Functions & Classes)
- **PatchSettings**: كلاس لتخزين إعدادات التلبيس الحاد، ADM scaler، controlnet softness، adaptive cfg.
- **patch_all()**: تطبيق جميع التلبيسات على وحدات ldm_patched.
- **calculate_weight_patched()**: دالة معدّلة لحساب أوزان LoRA مع دعم Fooocus format.
- **unet_forward_patched()**: دالة forward معدّلة للـ UNet تدعم Fooocus-style processing.
- **calc_cond_uncond_batch_patched()**: دالة حساب الشرط المشروط/غير المشروط مع dithering.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/async_worker.py, modules/default_pipeline.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/patch_clip.py, modules/patch_precision.py, modules/anisotropic.py, modules/inpaint_worker.py, modules/constants.py, ldm_patched/*