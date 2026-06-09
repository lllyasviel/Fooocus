---
### 📄 مسار الملف الافتراضي: docs/models/overview.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
هيكل مجلد النماذج (models/) يحتوي على جميع أنواع النماذج المستخدمة في Fooocus. كل مجلد فرعي يحتوي نوعاً محدداً من النماذج.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
**هيكل المجلدات:**
```
models/
├── checkpoints/       # نماذج SDXL الأساسية (.safetensors)
├── clip_vision/       # مشفرات CLIP البصرية
├── clip/              # مشفرات CLIP النصية
├── configs/           # ملفات تكوين النماذج (YAML)
├── controlnet/        # نماذج ControlNet (PyraCanny, CPDS)
├── diffusers/         # نماذج Diffusion
├── embeddings/        # Textual Inversion embeddings
├── gligen/            # نماذج GLIGEN grounding
├── hypernetworks/     # نماذج Hypernetwork
├── inpaint/           # نماذج Fooocus Inpaint (v1, v2.5, v2.6)
├── loras/             # LoRA adapters (أنماط, أداء, offset)
├── prompt_expansion/  # نموذج توسيع البرومبت Fooocus
├── safety_checker/    # نموذج كشف NSFW
├── sam/               # Segment Anything Model (vit_b, vit_l, vit_h)
├── style_models/      # نماذج مرجع الأنماط
├── unet/              # ملفات UNet
├── upscale_models/    # نماذج تكبير الصور
├── vae_approx/        # نماذج VAE التقريبية
└── vae/               # VAE decoders (SDXL, Pony, etc.)
```

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: لا يعتمد على ملفات أخرى
- **يُستخدم بواسطة**: modules/config.py (يُعرّف المسارات), modules/model_loader.py (يحمّل منها)
- **الأثر البرمجي**: غياب أي مجلد أو نموذج أساسي سيمنع Fooocus من العمل
