# وثيقة ملف: private_logger.md
## 1. الوظيفة الأساسية للملف
تسجيل الصور المولّدة وإنشاء معرض HTML. يحفظ الصور بالتنسيقات المختلفة (PNG, JPEG, WEBP) مع البيانات الوصفية، ويولّد ملف log.html لعرض النتائج.

## 2. الدوال والكلاسات (Functions & Classes)
- **log()**: دالة تسجيل الصورة الرئيسية، تحفظ الصورة مع البيانات الوصفية وتُرجع المسار.
- **get_current_html_path()**: الحصول على مسار ملف HTML الحالي.
- **log_cache**: قاموس تخزين مؤقت لبيانات السجل.

## 3. العتمادات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/async_worker.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/config.py, modules/flags.py, modules/meta_parser.py, modules/util.py, args_manager.py