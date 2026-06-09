---
### 📄 docs/extras/ip_adapter.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نقل ميزات الصورة إلى نموذج Diffusion عبر IP-Adapter. يُنفّذ آلية Cross-Attention محسّنة (Midjourney-style) تحقن ميزات CLIP image في طبقات attention لـ UNet. يدعم SD 1.x و SDXL مع نسخة Plus التي تستخدم Perceiver Resampler لتقليل عدد رموز الصورة إلى 16 استعلاماً ثابتاً.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class ImageProjModel(nn.Module)**: إسقاط تضمينات CLIP (768/1280/1664) إلى فضاء Cross-Attention (2048).
- **class To_KV(nn.Module)**: تعيين التضمينات المسقطة إلى أزواج key/value لجميع طبقات UNet attention.
- **class IPAdapterModel(nn.Module)**: يدمج ImageProjModel (أو Resampler للنسخة Plus) مع طبقات To_KV، يُحمّل الأوزان من safetensors.
- **sdp(keys, values, queries)**: غلاف للانتباه المُحسَّن (scaled dot-product memory-efficient attention).
- **load_ip_adapter(ckpt_path)**: تحميل CLIP vision model، التضمينات السالبة، وأوزان IP adapter. إرجاع (ip_adapter_model, clip_vision, clip_vision_negative).
- **clip_preprocess(image, size)**: تطبيع الصور لمدخل CLIP vision.
- **preprocess(clip_vision, ip_adapter, image, weight, dtype, device)**: تشغيل CLIP vision encoding، إسقاط الناتج عبر IP adapter، إرجاع condition tensors.
- **patch_model(model, ip_adapter, weight, start_at, end_at)**: استنساخ النموذج، ترقيع طبقات attention UNet بحقن key/value من IP adapter باستخدام تركيبة Midjourney. إرجاع النموذج المُرَقَّع.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: ldm_patched/modules/clip_vision.py, ldm_patched/modules/model_management.py, ldm_patched/ldm/modules/attention.py, extras/resampler.py (Resampler), ldm_patched/modules/model_patcher.py, modules/core.py, modules/ops.py, safetensors.torch
- **يُستدعى بواسطة**: modules/async_worker.py
- **الأثر البرمجي**: هذا الملف أساسي لميزة Image Prompt في Fooocus
