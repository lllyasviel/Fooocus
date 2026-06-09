---
### 📄 docs/fooocus_version.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
تعريف إصدار Fooocus الحالي (2.5.5) في متغير نصي واحد. يُستخدم في عنوان صفحة الويب، سجلات التوليد، والبيانات الوصفية للصور.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **version = '2.5.5'**: متغير نصي يحتوي على رقم الإصدار الحالي.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **لا يستدعي أي ملفات أخرى**
- **يُستدعى بواسطة**: launch.py (طباعة الإصدار), webui.py (عنوان الصفحة: `title = f'Fooocus {fooocus_version.version}'`), modules/async_worker.py (البيانات الوصفية للصور), modules/private_logger.py
- **الأثر البرمجي**: تغيير هذا المتغير يغيّر الإصدار المعروض في جميع أنحاء التطبيق والصور المُولّدة

**للمزيد من التفاصيل، راجع:** [FOOOCUS_TECHNICAL_DOC.md](FOOOCUS_TECHNICAL_DOC.md)
