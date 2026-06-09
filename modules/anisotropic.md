# وثيقة ملف: anisotropic.md
## 1. الوظيفة الأساسية للملف
تنفيذ ترشيح bilateral blur anisotropic. يوفر دوال لإنشاء نوى Gaussian وتطبيق الترشيح ثنائي الاتجاه.

## 2. الدوال والكلاسات (Functions & Classes)
- **gaussian()**: إنشاء نواة Gaussian.
- **get_gaussian_kernel1d()**: الحصول على نواة Gaussian أحادية البُعد.
- **get_gaussian_kernel2d()**: الحصول على نواة Gaussian ثنائية البُعد.
- **_bilateral_blur()**: تطبيق bilateral blur with guidance.
- **_compute_zero_padding()**: حساب التعبئة الصفرية.
- **_unpack_2d_ks()**: فتح حجم النواة ثنائية البُعد.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/patch.py
- الملفات التي يستدعيها هذا الملف لتشغيله: torch