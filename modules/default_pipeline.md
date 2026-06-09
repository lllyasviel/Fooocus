# وثيقة ملف: default_pipeline.md
## 1. الوظيفة الأساسية للملف
تنسيق وإدارة خط أنابيب المعالجة الافتراضي. يدير النماذج الأساسية والمرشحات، ويتحكم في تحميل النماذج وتحديثها، ويوفر واجهة لعمليات التوليد المختلفة (txt2img, img2img, inpaint, upscale).

## 2. الدوال والكلاسات (Functions & Classes)
- **model_base / model_refiner**: نموذج أساس ونموذج مرشح من نوع StableDiffusionModel.
- **refresh_base_model()**: تحميل/تحديث النموذج الأساسي.
- **refresh_refiner_model()**: تحميل/تحديث النموذج المرشح.
- **refresh_controlnets()**: إدارة وتحديث نماذج ControlNet المحملة.
- **assert_model_integrity()**: التحقق من توافق النموذج (SDXL فقط).
- **process_generate()**: تنفيذ عملية التوليد الرئيسية مع دعم جميع الوضعيات.
- **prepare_mask() / prepare_pixels()**: تحضير القناع والبكسلات للعملية.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/async_worker.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/core.py, modules/patch.py, modules/config.py, modules/flags.py, modules/inpaint_worker.py, extras/expansion.py, extras/vae_interpose.py