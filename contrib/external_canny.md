# وثيقة ملف: external_canny.py
## 1. الوظيفة الأساسية للملف
يُنفذ خوارزمية كشف الحواف Canny لاستخراج ميزات الصور، تُستخدم للتحكم في نماذج التوليد عبر ControlNet.

## 2. الدوال والكلاسات (Functions & Classes)
- get_canny_nms_kernel(): دالة لإرجاع أنوية القمع غير الأقصى لخوارزمية Canny
- get_hysteresis_kernel(): دالة لإرجاع أنوية الارتجاع لخوارزمية Canny
- gaussian_blur_2d(): دالة لتطبيق ضبابية غاوسية ثنائية الأبعاد
- canny(): دالة رئيسية لكشف الحواف باستخدام خوارزمية Canny
- CannyEdgePreprocessor: كلاس لمعالجة الصور واستخراج حواف Canny

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - torch.nn.functional
  - ldm_patched/modules/model_management.py