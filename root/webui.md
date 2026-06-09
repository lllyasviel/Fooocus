---
### 📄 مسار الملف الافتراضي: docs/root/webui.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
بناء واجهة المستخدم الرسومية الكاملة باستخدام Gradio. يحدد كل عناصر التحكم في الواجهة (المدخلات، الأزرار، القوائم، التبويبات) ويربطها بالدوال البرمجية المناسبة. يمثل الجزء الأكبر من كود الواجهة (1128 سطر).

المهام الرئيسية:
- بناء التخطيط الرئيسي: عمود معاينة + عمود إعدادات
- تبويبات الصورة المدخلة: Upscale/Variation, Image Prompt, Inpaint/Outpaint, Describe, Enhance, Metadata
- تبويبات الإعدادات: Settings, Styles, Models, Advanced
- ربط الأحداث: تغيير الإعدادات، تحميل الإعدادات المسبقة، بدء التوليد
- معالجة النتائج: عرض التقدم، عرض الصور النهائية، حفظ السجلات

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **get_task(*args)**: تحويل مدخلات UI إلى كائن AsyncTask.
- **generate_clicked(task)**: الحلقة الرئيسية لتوليد الصور - تتابع التقدم وتعرض النتائج.
- **sort_enhance_images(images, task)**: ترتيب صور التحسين حسب إحصائيات التحسين.
- **inpaint_mode_change(mode, inpaint_engine_version)**: تحديث واجهة التحكم عند تغيير وضع Inpaint.
- **trigger_describe(modes, img, apply_styles)**: وصف الصورة المدخلة باستخدام BLIP أو WD14.
- **trigger_metadata_import(file, state_is_generating)**: استيراد المعاملات من صورة Fooocus.
- **load_data_outputs**: قائمة مخرجات تحميل الإعدادات (~50 عنصر UI).
- **ctrls**: قائمة مدخلات المهمة (~100 عنصر UI).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: modules/config.py, modules/async_worker.py, modules/flags.py, modules/html.py, modules/gradio_hijack.py, modules/style_sorter.py, modules/meta_parser.py, modules/sdxl_styles.py, modules/private_logger.py, modules/ui_gradio_extensions.py, modules/auth.py, modules/util.py, modules/constants.py, extras/inpaint_mask.py, ldm_patched/modules/model_management.py, args_manager.py, shared.py, launch.py, fooocus_version.py
- **يُستدعى بواسطة**: launch.py (يُستورد في نهايته via `from webui import *`)
- **الأثر البرمجي**: أي تعديل على العناصر هنا سيتغير مباشرة في واجهة المستخدم
