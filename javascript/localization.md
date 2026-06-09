---
### 📄 مسار الملف الافتراضي: docs/javascript/localization.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نظام التدويل/التوطين (153 سطر). يحمّل قاموس window.localization ويترجم عناصر النص في DOM.

الميزات: ترجمة النصوص، عناوين titles، حقول placeholder، تلميحات tooltips، دعم RTL.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- ترجمة عقد النص في DOM باستخدام القاموس.
- حفظ النص الأصلي في data-original-text.
- دعم RTL عبر media queries CSS.
- **refresh_style_localization()**: تحديث ترجمة الأنماط.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: DOM API, window.localization (من modules/localization.py)
- **يُستخدم بواسطة**: webui.py (عبر _js callbacks)
- **الأثر البرمجي**: تغيير هنا يؤثر على ترجمة جميع عناصر الواجهة
