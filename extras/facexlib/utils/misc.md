---
### 📄 docs/extras/facexlib/utils/misc.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
دوال مساعدة متنوعة: تحميل الملفات من URLs مع دعم إعادة التوجيه (redirect)، إنشاء المجلدات، وإدارة مسارات التحميل. يُستخدم بشكل أساسي لتحميل أوزان النماذج (FaceRestoreHelper.pth، RetinaFace.pth، إلخ) في facexlib.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **load_file_from_url(url, model_dir=None, progress=True, file_name=None)**: تحميل ملف من URL. يتحقق من ملفات التحميل المسبق، يُنشئ المجلدات، ويُعيد المسار المحلي.
- **buffered_touch(file_path, times=None)**: تحديث وقت تعديل الملف.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: os, os.path as osp, requests, torch.hub
- **يُستدعى بواسطة**: extras/facexlib/utils/__init__.py, extras/facexlib/utils/face_restoration_helper.py
- **الأثر البرمجي**: أساسي لتحميل أوزان جميع نماذج facexlib
