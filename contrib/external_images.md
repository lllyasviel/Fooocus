# وثيقة ملف: external_images.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في معالجة الصور مثل القص والتغيير الحجم والتحويل.

## 2. الدوال والكلاسات (Functions & Classes)
- ImageCrop: كلاس لقص الصور بأبعاد محددة
- RepeatImageBatch: كلاس لتكرار صورة لإنشاء مجموعة
- ResizeImage: كلاس لتغيير حجم الصور
- ImageScale: كلاس لتكبير/تصغير الصور باستخدام طرق مختلفة
- ImageInvert: كلاس لعكس ألوان الصور
- ImageBatch: كلاس لإنشاء مجموعة من الصور
- ImagePadForOutpaint: كلاس لتحضير الصور لل-outpainting
- ImageFromBatch: كلاس لاستخراج صورة من مجموعة

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/contrib/external.py
  - ldm_patched/utils/path_utils.py
  - ldm_patched/modules/args_parser.py
  - PIL
  - numpy
  - json
  - os