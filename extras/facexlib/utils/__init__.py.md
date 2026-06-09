# وثيقة ملف: __init__.py.md
## 1. الوظيفة الأساسية للملف
ملف التهيئة لوحدة الأدوات المساعدة، يوفر الدوال الأساسية للتعامل مع الوجوه.

## 2. الدوال والكلاسات (Functions & Classes)
- **align_crop_face_landmarks()**: محاذاة وقص الوجه بناءً على نقاط المعالم
- **compute_increased_bbox()**: حساب صندوق التحديد الموسّع
- **get_valid_bboxes()**: الحصول على صناديق التحديد الصالحة
- **paste_face_back()**: لصق الوجه المستعاد في الصورة الأصلية
- **img2tensor()**: تحويل الصورة إلى tensor
- **load_file_from_url()**: تحميل ملف من URL
- **scandir()**: فحص المجلد

## 3. ال依赖يات والعلاقات
- الملفات التي تستدعي هذا الملف: extras/facexlib/detection/__init__.py, extras/facexlib/parsing/__init__.py, extras/facexlib/utils/face_restoration_helper.py
- الملفات التي يستدعيها هذا الملف لتشغيله: extras/facexlib/utils/face_utils.py, extras/facexlib/utils/misc.py
