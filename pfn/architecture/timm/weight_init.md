# وثيقة ملف: weight_init.py
## 1. الوظيفة الأساسية للملف
يحتوي على دوال تهيئة الأوزان لبناء نماذج مختلفة.

## 2. الدوال والكلاسات (Functions & Classes)
- trunc_normal_(): دالة لتهيئة التوزيع الطبيعي المقطوع
- trunc_normal_stddev(): دالة لحساب الانحراف المعياري للتوزيع الطبيعي المقطوع
- lecun_normal_(): دالة لتهيئة LeCun الطبيعية
- constant_init(): دالة لتهيئة بقيمة ثابتة
- normal_init(): دالة لتهيئة بالتوزيع الطبيعي
- xavier_uniform_(): دالة لتهيئة Xavier uniform
- variance_scaling_(): دالة لتهيئة تباين التحجيم

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/pfn/architecture/DAT.py
  - ldm_patched/pfn/architecture/HAT.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch