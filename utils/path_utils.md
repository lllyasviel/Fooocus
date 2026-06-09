# وثيقة ملف: path_utils.py
## 1. الوظيفة الأساسية للملف
يحتوي على دوال مساعدة لإدارة مسارات الملفات والمجلدات.

## 2. الدوال والكلاسات (Functions & Classes)
- get_full_path(): دالة للحصول على المسار الكامل لملف
- get_filename_list(): دالة للحصول على قائمة أسماء الملفات في مجلد
- get_folder_paths(): دالة للحصول على مسارات المجلدات
- add_model_folder_path(): دالة لإضافة مسار مجلد نموذج جديد

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/contrib/external.py
  - ldm_patched/contrib/external_images.py
  - ldm_patched/contrib/external_hypernetwork.py
  - ldm_patched/contrib/external_model_merging.py
  - ldm_patched/contrib/external_video_model.py
  - ldm_patched/contrib/external_upscale_model.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - os
  - json