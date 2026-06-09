# وثيقة ملف: sample_hijack.md
## 1. الوظيفة الأساسية للملف
اختطاف وتعديل عملية أخذ العينات لدعم التبديل بين النموذج الأساسي والمرشح. يوفر دوال CLIP منفصلة وعملية عينات معدّلة تدعم refiner switching.

## 2. الدوال والكلاسات (Functions & Classes)
- **clip_separate()**: فصل مخرجات CLIP للتكيف مع نماذج مختلفة (SDXL, SDXLRefiner, SD1.5).
- **clip_separate_inner()**: دالة داخلية لفصل المخرجات حسب نوع النموذج المستهدف.
- **clip_separate_after_preparation()**: فصل بعد تحضير الشروط.
- **sample_hacked()**: دالة العينات المعدّلة التي تدعم التبديل بين base وrefiner أثناء عملية التوليد.
- **current_refiner / refiner_switch_step**: متغيرات عامة لتتبع حالة المرشح الحالي.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/default_pipeline.py, modules/core.py
- الملفات التي يستدعيها هذا الملف لتشغيله: ldm_patched/modules/samplers.py, ldm_patched/modules/sample.py, ldm_patched/modules/model_base.py, ldm_patched/k_diffusion/sampling.py