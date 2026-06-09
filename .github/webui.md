---
### 📄 docs/webui.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
بناء واجهة المستخدم الرسومية الكاملة باستخدام Gradio (1128 سطر). يُعرّف التخطيط الرئيسي مع عمودين (معاينة + إعدادات)، 6 تبويبات للصورة المدخلة (Upscale, Image Prompt, Inpaint, Describe, Enhance, Metadata)، و 4 تبويبات إعدادات (Settings, Styles, Models, Advanced). يربط جميع عناصر التحكم بالدوال البرمجية المناسبة.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **get_task(*args)**: تحويل ~100 مدخل UI إلى كائن AsyncTask.
- **generate_clicked(task)**: الحلقة الرئيسية لتوليد الصور. تتابع progress (preview/results/finish)، تدعم skip المَعاينة للاتصالات البطيئة، وتُرتب صور التحسين.
- **sort_enhance_images(images, task)**: ترتيب صور التحسين حسب إحصائيات enhance_stats.
- **inpaint_mode_change(mode, inpaint_engine_version)**: تحديث واجهة Inpaint عند تغيير الوضع (Detail/Modify/Outpaint).
- **stop_clicked(currentTask)**: إيقاف التوليد الحالي.
- **skip_clicked(currentTask)**: تخطي التوليد الحالي.
- **generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug)**: توليد قناع Inpaint من الصورة.
- **trigger_metadata_preview(file)**: قراءة البيانات الوصفية من صورة Fooocus.
- **trigger_describe(modes, img, apply_styles)**: وصف الصورة باستخدام BLIP (Photo) أو WD14 (Anime).
- **trigger_auto_describe(mode, img, prompt, apply_styles)**: وصف تلقائي عند رفع صورة.
- **dev_mode_checked(r)**: إظهار/إخفاء أدوات المطور.
- **refresh_files_clicked()**: تحديث قوائم النماذج والـ LoRA و VAE.
- **preset_selection_change(preset, is_generating, inpaint_mode)**: تحميل وتطبيق القالب المحدد.
- **parse_meta(raw_prompt_txt, is_generating)**: تحليل JSON من حقل البرومبت.
- **trigger_metadata_import(file, state_is_generating)**: استيراد المعاملات من صورة.
- **dump_default_english_config()**: تصدير ملف الترجمة الإنجليزية (مُعلّق).

المتغيرات: title, currentTask (State), gallery (Grad Gallery), load_data_outputs (قائمة مخرجات ~50 عنصر), ctrls (قائمة مدخلات ~100 عنصر).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: modules/config.py, modules/async_worker.py, modules/flags.py, modules/html.py, modules/gradio_hijack.py, modules/style_sorter.py, modules/meta_parser.py, modules/sdxl_styles.py, modules/private_logger.py, modules/ui_gradio_extensions.py, modules/auth.py, modules/util.py, modules/constants.py, extras/inpaint_mask.py, ldm_patched/modules/model_management.py, args_manager.py, shared.py, launch.py, fooocus_version.py
- **يُستدعى بواسطة**: launch.py (via `from webui import *`)
- **الأثر البرمجي**: أي تغيير على عناصر UI، الأزرار، التبويبات، أو ربط الأحداث يؤثر مباشرة على واجهة المستخدم

**للمزيد من التفاصيل، راجع:**
- [shared.md](shared.md) - المتغيرات المشتركة
- [MODULES.md](MODULES.md) - الوحدات الأساسية
