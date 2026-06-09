---
### 📄 مسار الملف الافتراضي: docs/javascript/zoom.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نظام تكبير/تحريك (646 سطر) لأداة Inpaint Canvas. يدعم اختصارات لوحة المفاتيح وعرض الصور الطويلة تلقائياً.

الاختصارات: Shift+wheel=تكبير، Ctrl+wheel=حجم الفرش، R=إعادة تعيين، S=ملء الشاشة، F=وضع التحريك.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **applyZoomAndPan()**: تطبيق نظام التكبير/التحريك على #inpaint_canvas و #inpaint_mask_canvas.
- **viewer_to_top()/viewer_to_bottom()**: مساعدة التمرير.
- **cancelGenerateForever()**: إلغاء التوليد الم forever.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: Canvas API, DOM
- **يُستخدم بواسطة**: webui.py (عبر _js callbacks)
- **الأثر البرمجي**: تغيير هنا يؤثر على تجربة رسم Inpaint
