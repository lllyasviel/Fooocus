# وثيقة ملف: lora.md
## 1. الوظيفة الأساسية للملف
تحميل وتطبيق نماذج LoRA (Low-Rank Adaptation) مع إدارة تعيين المفاتيح.

## 2. الدوال والكلاسات (Functions & Classes)
- `load_lora()`: تحميل نموذج LoRA
- `apply_lora()`: تطبيق LoRA على النموذج
- `map_lora_keys()`: تعيين مفاتيح LoRA
- `merge_lora()`: دمج LoRA مع النموذج
- `calculate_lora_strength()`: حساب قوة LoRA
- `get_lora_params()`: الحصول على معلمات LoRA

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `model_patcher.py`, `sd.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `ops.py`, `utils.py`
