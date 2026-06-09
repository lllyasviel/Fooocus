---
### 📄 مسار الملف الافتراضي: docs/css/style.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف التنسيق الرئيسي لواجهة Fooocus (418 سطر). يُحوّل واجهة Gradio الافتراضية إلى تصميم Fooocus المخصص. مستوحى من AUTOMATIC1111 Stable Diffusion WebUI.

الميزات الرئيسية:
- شريط تحميل وشريط تقدم (لون أزرق #3498db)
- تنسيق محرر البرومبت ونسبة العرض والارتفاع
- قائمة السياق (right-click menus) مع حدود برتقالية #a55000
- نظام تلميحات Canvas مع أيقونات معلومات
- عارض صور Lightbox بملء الشاشة مع تنقل prev/next
- لوحة اختيار الأنماط المُمرّرة بتنسيق عمودين
- طبقة معاينة النمط التي تتبع المؤشر
- تنسيق قلم Inpaint canvas

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- لا يحتوي على دوال JavaScript - ملف CSS فقط
- الفئات الرئيسية: .loading, .progress-bar, #positive_prompt, .context-menu, .canvas-tooltip, .lightbox-modal, .style_selections, .main_view, .final_gallery

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: لا يعتمد على ملفات أخرى
- **يُستخدم بواسطة**: modules/ui_gradio_extensions.py (يحقنه في الواجهة)
- **الأثر البرمجي**: تغيير هنا سيتغير مظهر الواجهة بالكامل
