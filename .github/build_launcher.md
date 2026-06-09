---
### 📄 docs/build_launcher.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
إنشاء ملفات batch (.bat) على Windows لتشغيل Fooocus بسهولة. يتحقق من وجود مجلد python_embeded (البرمجية المدمجة) ويُنشئ 3 ملفات تشغيل: run.bat (افتراضي)، run_anime.bat، run_realistic.bat.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **build_launcher()**: التحقق من بيئة Windows standalone build (وجود python_embeded). إنشاء ملفات .bat لكل قالب: None (افتراضي)، 'anime'، 'realistic'. كل ملف batch يستدعي entry_with_update.py مع المعاملات المناسبة.
- **win32_cmd**: قالب نص batch يستدعي python_embeded/python.exe مع entry_with_update.py.
- **is_win32_standalone_build**: متغير منطقي يتحقق من وجود python_embeded.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **لا يستدعي ملفات أخرى غير os**
- **يُستدعى بواسطة**: launch.py (السطر 76: `build_launcher()`)
- **الأثر البرمجي**: وظيفته مساعدة فقط - يُنشئ ملفات batch لراحة المستخدم. لا يؤثر على الوظيفة الرئيسية

**للمزيد من التفاصيل، راجع:**
- [launch.md](launch.md) - نقطة التشغيل الرئيسية
- [entry_with_update.md](entry_with_update.md) - نقطة الدخول مع التحديث
