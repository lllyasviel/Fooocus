# وثيقة ملف: external_compositing.py
## 1. الوظيفة الأساسية للملف
يُنفذ عمليات التجميع (compositing) للصور والـ masks باستخدام خوارزميات Porter-Duff المختلفة.

## 2. الدوال والكلاسات (Functions & Classes)
- resize_mask(): دالة لتغيير حجم الـ mask باستخدام الاستيفاء الخطي
- PorterDuffMode: تعداد (enum) بأنماط التجميع المختلفة مثل ADD وCLEAR وMULTIPLY
- porter_duff_composite(): دالة لتطبيق تجميع Porter-Duff على الصور
- Composite: كلاس لتجميع صورتين معًا باستخدام أنماط مختلفة
- PorterDuffComposite: كلاس لتطبيق تجميع Porter-Duff على صورتين

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - numpy
  - ldm_patched/modules/utils.py