---
### 📄 docs/entry_with_update.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نقطة الدخول الفعلية للتطبيق مع دعم التحديث التلقائي من Git. يستخدم pygit2 لفتح المستودع المحلي، يتصل بالremote (origin)، يُحضّر أحدث التغييرات، ثم يُنفّذ Fast-Forward merge إذا كان الفرع المحلي متأخراً. بعد نجاح التحديث، يستورد launch.py.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
لا يحتوي على دوال أو كلاسات - الكود يُنفّذ في المستوى العلوي (module level):
- فتح المستودع عبر pygit2.Repository
- جلب (fetch) من remote origin
- تحليل حالة الدمج (merge_analysis): up-to-date / fastforward / normal
- في حالة Fast-Forward: تعيين target الـ local branch، checkout، reset hard
- في حالة Normal: طباعة رسالة فشل (تعديلات محلية)
- استيراد launch.py في النهاية

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: pygit2, launch.py
- **يُستدعى بواسطة**: run.bat / run_anime.bat / run_realistic.bat (ملفات التشغيل على Windows)
- **الأثر البرمجي**: هذا الملف هو نقطة البداية الفعلية - أي خطأ هنا سيمنع التطبيق من التشغيل
