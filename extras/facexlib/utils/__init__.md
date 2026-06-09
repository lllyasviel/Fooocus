---
### 📄 docs/extras/facexlib/utils/__init__.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف تهيئة حزمة utils في facexlib. يُصدّر الدوال المساعدة العامة: تحميل النماذج من الملفات، و FaceRestoreHelper لإدارة دورة حياة استعادة الوجه الكاملة (كشف + محاذاة + استعادة + لصق).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **load_file_from_url(url, model_dir, progress, file_name)**: تحميل ملف من URL (مع آلية إعادة توجيه/download)، حفظه في model_dir، إرجاع المسار المحلي.
- **class FaceRestoreHelper**: فئة إدارة استعادة الوجه الكاملة.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/facexlib/utils/misc.py, extras/facexlib/utils/face_restoration_helper.py, urllib, os
- **يُستدعى بواسطة**: extras/face_crop.py, extras/facexlib/utils/face_restoration_helper.py
- **الأثر البرمجي**: نقطة الوصول لأدوات استعادة الوجه
