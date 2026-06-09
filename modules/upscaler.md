# وثيقة ملف: upscaler.md
## 1. الوظيفة الأساسية للملف
تنفيذ عملية تكبير الصور باستخدام نموذج ESRGAN. يوفر دالة perform_upscale لتكبير الصور بمعامل 4x.

## 2. الدوال والكلاسات (Functions & Classes)
- **perform_upscale()**: دالة تقوم بتكبير الصورة باستخدام نموذج ESRGAN المحمل.
- **opImageUpscaleWithModel**: كائن عملية التكبير من ldm_patched.
- **model**: النموذج المحمل (ESRGAN) أو None إذا لم يتم تحميله بعد.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/inpaint_worker.py, modules/async_worker.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/core.py, modules/config.py, ldm_patched/contrib/external_upscale_model.py, ldm_patched/pfn/architecture/RRDB.py