# وثيقة ملف: sd.md
## 1. الوظيفة الأساسية للملف
تحميل وتخزين نماذج الاختراق (Checkpoints) ونماذج التخزين المؤقت (CKPT/Safetensors) مع دعم الترميز والفك الترميز للبيانات المخزنة.

## 2. الدوال والكلاسات (Functions & Classes)
- `load_checkpoint_guess_config()`: تحميل نقطة التحقق مع تخمين الإعدادات تلقائياً
- `load_state_dict()`: تحميل حالة النموذج من ملف
- `save_checkpoint()`: حفظ نقطة التحقق
- `create_checkpoint()`: إنشاء نقطة تحقق جديدة
- `get_state_dict()`: الحصول على حالة النموذج
- `disable_weight_init()`: تعطيل تهيئة الأوزان
- `enable_weight_init()`: تمكين تهيئة الأوزان

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `model_management.py`, `model_detection.py`, `sample.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_base.py`, `model_patcher.py`, `ops.py`
