# وثيقة ملف: config.md
## 1. الوظيفة الأساسية للملف
إدارة إعدادات التطبيق وتهيئة المسارات. يقرأ ملفات الإعدادات (config.txt، presets)، ويحدد مسارات النماذج والملفات المختلفة، ويقدم واجهة للحصول على الإعدادات المطلوبة.

## 2. الدوال والكلاسات (Functions & Classes)
- **get_config_path()**: تحصل على مسار الإعدادات من المتغيرات البيئية أو المسار الافتراضي.
- **try_load_deprecated_user_path_config()**: تتعامل مع ملفات الإعدادات القديمة وتوحيد المسارات.
- **default_*()**: دوال للحصول على القيم الافتراضية للمعلمات المختلفة.
- **downloading_*()**: دوال لتحميل النماذج عند عدم وجودها محلياً.
- **paths_*()**: مسارات مجلدات النماذج (checkpoints, loras, embeddings, vae, controlnet, etc).
- **default_loras()**: إعدادات LoRA الافتراضية.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: جميع ملفات modules تقريباً
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/model_loader.py, modules/extra_utils.py, modules/flags.py, modules/sdxl_styles.py, args_manager.py