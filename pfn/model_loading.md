# وثيقة ملف: model_loading.md
## 1. الوظيفة الأساسية للملف
يحتوي على دوال لتحميل نماذج PyTorch و معالجة state dicts.

## 2. الدوال والكلاسات (Functions & Classes)
- load_state_dict(): دالة لتحميل state dict من ملف
- load_torch_file(): دالة لتحميل ملف PyTorch
- safe_load(): دالة لتحميل آمن لملفات PyTorch

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/contrib/external_upscale_model.py
  - ldm_patched/contrib/external_hypernetwork.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - torch
  - safetensors.torch