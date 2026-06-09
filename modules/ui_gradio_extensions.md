# وثيقة ملف: ui_gradio_extensions.md
## 1. الوظيفة الأساسية للملف
حقن JavaScript و CSS في واجهة Gradio. يعدل استجابات القالب لإضافة ملفات JS و CSS مخصصة.

## 2. الدوال والكلاسات (Functions & Classes)
- **javascript_html()**: إنشاء وسوم script لملفات JavaScript.
- **css_html()**: إنشاء وسوم link لملفات CSS.
- **reload_javascript()**: تعديل استجابات القالب لإضافة JS/CSS.
- **webpath()**: تحويل مسار ملف إلى مسار ويب.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: webui.py
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/localization.py, args_manager.py, gradio.routes