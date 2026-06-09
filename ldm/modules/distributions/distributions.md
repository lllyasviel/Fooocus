# وثيقة ملف: distributions.md
## 1. الوظيفة الأساسية للملف
يحتوي على كلاس DiagonalGaussianDistribution للتوزيع الغاوسي القطري المستخدم في Autoencoders.

## 2. الدوال والكلاسات (Functions & Classes)
- DiagonalGaussianDistribution: كلاس للتوزيع الغاوسي القطري
  - __init__(): دالة التهيئة مع المعلمات mean وlogvar
  - sample(): دالة لأخذ عينات من التوزيع
  - mode(): دالة للحصول على الوضع (أكبر احتمال)
  - kl(): دالة لحساب تباعد Kullback-Leibler
  - nll(): دالة لحساب السالب للإحتمال اللوغاريتمي
  - mean: خاصية للحصول على المتوسط
  - variance: خاصية للحصول على التباين

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/ldm/models/autoencoder.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch