# وثيقة ملف: meta_parser.md
## 1. الوظيفة الأساسية للملف
تحليل وحفظ البيانات الوصفية (metadata) للصور. يدعم تنسيقين: Fooocus (JSON) و A1111 (نص عادي). يوفر دوال لاستخراج المعلمات من الصور وحفظها.

## 2. الدوال والكلاسات (Functions & Classes)
- **MetadataParser**: كلاس أساسي (ABC) لتحليل البيانات الوصفية.
- **load_parameter_button_click()**: تحميل المعلمات من البيانات الوصفية.
- **get_str() / get_number() / get_list()**: دوال مساعدة لاستخراج قيود بأنواع مختلفة.
- **get_exif()**: إنشاء بيانات EXIF من المعلمات.
- **re_param / re_imagesize**: أنماط regex لتحليل النص.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/private_logger.py, webui.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/config.py, modules/sdxl_styles.py, modules/flags.py, modules/hash_cache.py, modules/util.py, fooocus_version.py