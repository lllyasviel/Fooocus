---
### 📄 مسار الملف الافتراضي: docs/root/build_launcher.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
إنشاء ملفات batch (.bat) على نظام Windows لتشغيل Fooocus بسهولة. يتحقق من وجود مجلد python_embeded (البرمجية المدمجة) ويُنشئ ملفات تشغيل لكل قالب: run.bat (افتراضي)، run_anime.bat، run_realistic.bat.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **build_launcher()**: تحقق من بيئة Windows standalone build، وتُنشئ ملفات batch لكل قالب.
- **win32_cmd**: نص قالب batch الذي يستدعي python_embeded/python.exe مع entry_with_update.py.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: لا يعتمد على ملفات أخرى
- **يُستدعى بواسطة**: launch.py (يُنادى في بداية التشغيل)
- **الأثر البرمجي**: لا يؤثر على الوظيفة الرئيسية - فقط يُنشئ ملفات تشغيل مريحة
