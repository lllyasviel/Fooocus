# وثيقة ملف: model_patcher.md
## 1. الوظيفة الأساسية للملف
تطبيق تعديلات LoRA والتصحيحات على النماذج المحملة مع إدارة التخزين المؤقت للتعديلات.

## 2. الدوال والكلاسات (Functions & Classes)
- `ModelPatcher()`: كلاس رئيسي لتطبيق التصحيحات على النماذج
- `add_patches()`: إضافة patches للنموذج
- `remove_patches()`: إزالة patches من النموذج
- `load()`: تحميل النموذج
- `unload()`: تفريغ النموذج
- `clone()`: استنساخ النموذج
- `get_model()`: الحصول على النموذج
- `set_model_attn1_patch()`: تعيين patch للانتباه
- `set_model_patch()`: تعيين patch عام

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd.py`, `samplers.py`, `model_management.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `lora.py`, `ops.py`, `model_base.py`
