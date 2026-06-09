# وثيقة ملف: core.md
## 1. الوظيفة الأساسية للملف
النواة الأساسية لنظام Stable Diffusion. يحتوي على كلاس StableDiffusionModel الذي يمثل النموذج الكامل (UNet, VAE, CLIP)، ويوفر دوال لتحميل النماذج وتطبيق LoRA وتنفيذ عملية العينات (sampling).

## 2. الدوال والكلاسات (Functions & Classes)
- **StableDiffusionModel**: كلاس يمثل نموذج Stable Diffusion الكامل مع UNet و VAE و CLIP و CLIP Vision.
  - **refresh_loras()**: تحديث وتطبيق أوزان LoRA على النموذج.
- **load_model()**: تحميل نموذج checkpoint مع تحديد VAE مخصص اختياري.
- **load_controlnet()**: تحميل نموذج ControlNet.
- **ksampler()**: تنفيذ عملية العينات باستخدام k-sampler مع دعم refiner.
- **decode_latent_VAE() / encode_latent_VAE()**: تشفير وفك تشفير الفضاء الكامن.
- **numpy_to_pytorch() / pytorch_to_numpy()**: تحويل بين تنسيقات الصور.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: modules/default_pipeline.py, modules/upscaler.py, modules/async_worker.py
- الملفات التي يستدعيها هذا الملف لتشغيله: ldm_patched/modules/*, modules/lora.py, modules/util.py, modules/config.py, modules/sample_hijack.py