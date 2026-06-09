# وثيقة ملف: inpaint_worker.md
## 1. الوظيفة الأساسية للملف
إدارة عملية التلوين (inpainting). يحتوي على كلاس InpaintHead لمعالجة القناع، ودوال مساعدة لمعالجة الصور والقناع باستخدام fil morphological operations.

## 2. الدوال والكلاسات (Functions & Classes)
- **InpaintHead**: كلاس PyTorch يمثل طبقة رأس التلوين.
- **current_task**: متغير عالمي يمثل المهمة الحالية.
- **box_blur()**: تطبيق ضباب صندوكي.
- **max_filter_opencv()**: تطبيق فلتر الأقصى باستخدام OpenCV.
- **morphological_open()**: عملية morphological لتنعيم القناع.
- **compute_initial_abcd()**: حساب الإحداثيات الأولية للمنطقة المعالجة.
- **regulate_abcd()**: ضمان أن الإحداثيات ضمن حدود الصورة.
- **up255()**: تحويل القيم إلى 0 أو 255.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/default_pipeline.py, modules/patch.py, modules/async_worker.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/util.py, modules/upscaler.py, cv2 (OpenCV)