# وثيقة ملف: external_hypernetwork.py
## 1. الوظيفة الأساسية للملف
يُنفذ دعم Hypernetworks لتعديل نماذج التوليد باستخدام شبكات عصبية فرعية.

## 2. الدوال والكلاسات (Functions & Classes)
- load_hypernetwork_patch(): دالة لتحميل تطبيق Hypernetwork من ملف
- HypernetworkLoader: كلاس لتحميل وتطبيق Hypernetwork على النموذج

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/modules/utils.py
  - ldm_patched/utils/path_utils.py
  - torch