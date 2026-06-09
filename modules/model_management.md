# وثيقة ملف: model_management.md
## 1. الوظيفة الأساسية للملف
إدارة ذاكرة GPU وVRAM، تحميل وتفريغ النماذج، وتنسيق موارد الحوسبة بين الأجهزة.

## 2. الدوال والكلاسات (Functions & Classes)
- `load_model_gpu()`: تحميل النموذج على GPU
- `free_memory()`: تحرير الذاكرة
- `get_torch_device()`: الحصول على جهاز PyTorch
- `get_vram_total()`: الحصول على إجمالي VRAM
- `get_free_vram()`: الحصول على VRAM المتاح
- `cleanup_models()`: تنظيف النماذج المحملة
- `unload_all_models()`: تفريغ جميع النماذج
- `thread_pool()`: مجموعة خيوط للمعالجة بالتوازي

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sd.py`, `samplers.py`, جميع ملفات النماذج
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_patcher.py`, مكتبات PyTorch
