# وثيقة ملف: localization.md
## 1. الوظيفة الأساسية للملف
إدارة الترجمة ودعم اللغة. يقرأ ملفات الترجمة من مجلد language ويوفر دوال لإنشاء JavaScript للترجمة.

## 2. الدوال والكلاسات (Functions & Classes)
- **localization_js()**: إنشاء JavaScript لتطبيق الترجمة في واجهة الويب.
- **dump_english_config()**: تصدير جميع النصوص الإنجليزية لملف ترجمة.
- **current_translation**: قاموس الترجمة الحالي.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/ui_gradio_extensions.py, modules/style_sorter.py
- الملفات التي يستدعيها هذا الملف لتشغيله: json