---
### 📄 مسار الملف الافتراضي: docs/javascript/contextMenus.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نظام قائمة السياق بالنقر الماوس الأيمن (166 سطر) من A1111. يوفر API لإضافة/حذف خيارات القائمة.

الميزة الرئيسية: خيار "Generate forever" على زر التوليد (يعيد التوليد كل 500ms).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **appendContextMenuOption(targetElementSelector, text, callback)**: إضافة خيار للقائمة.
- **removeContextMenuOption(targetElementSelector, text)**: حذف خيار من القائمة.
- تحديد موضع القائمة ديناميكياً للبقاء ضمن حدود النافذة.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: DOM, MouseEvent
- **يُستخدم بواسطة**: webui.py
- **الأثر البرمجي**: تغيير هنا يؤثر على قوائم السياق
