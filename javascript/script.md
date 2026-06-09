---
### 📄 مسار الملف الافتراضي: docs/javascript/script.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
إطار عمل واجهة المستخدم الأساسي (263 سطر). يوفر دوال مساعدة لـ Gradio ونظام مراقبة تغييرات DOM.

الوظائف الرئيسية:
- gradioApp() - العثور على عنصر تطبيق Gradio
- نظام تسجيل الارتجاعات: onUiUpdate, onUiLoaded, onUiTabChange, onOptionsChanged
- MutationObserver لتغييرات DOM
- طبقة معاينة الأنماط - عرض صورة مصغرة عند التمرير فوق مربعات الاختيار
- اختصار Ctrl+Enter لبدء التوليد
- معالجة أخطاء الاتصال (إظهار زر إعادة الاتصال)

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **gradioApp()**: إرجاع عنصر DOM الرئيسي لـ Gradio.
- **onUiUpdate(callback)**: تسجيل ارتجاع عند تحديث UI.
- **onUiLoaded(callback)**: تسجيل ارتجاع عند تحميل UI.
- **uiElementIsVisible(el)**: التحقق من ظهور عنصر UI.
- **playNotification()**: تشغيل صوت الإشعار.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: DOM API, MutationObserver
- **يُستخدم بواسطة**: جميع ملفات JS الأخرى تعتمد عليه
- **الأثر البرمجي**: تغيير هنا يؤثر على سلوك جميع مكونات UI التفاعلية
