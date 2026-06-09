---
### 📄 docs/extras/safety_checker/models/safety_checker.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج فحص أمان NSFW مخصص (ليس من transformers). يُنفّذ StableDiffusionSafetyChecker مخصّصاً مع CLIP image encoder (VisionTransformer) + ConceptClassifier (3 طبقات خطية) لتصنيف الصور إلى فئات NSFW. يُستخدم كبديل لنموذج transformers عندما لا يتوفر نموذج الـ safety checker الافتراضي.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class VisionTransformer(nn.Module)**: CLIP image encoder مخصّص (ViT-B/32): PatchEmbed + TransformerBlocks + LayerNorm.
- **class ConceptClassifier(nn.Module)**: مصنّف NSFW: Linear(768,512) + GELU + Linear(512,256) + GELU + Linear(256, concept_tokens).
- **class StableDiffusionSafetyChecker(nn.Module)**: النموذج الكامل: ViT + ConceptClassifier.
  - `forward(self, clip_input)`: تشغيل الصورة عبر ViT ثم المصنِّف، إرجاع logits.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn, collections
- **يُستدعى بواسطة**: extras/censor.py (Censor.init - تحميل النموذج)
- **الأثر البرمجي**: بديل فحص NSFW عند عدم توفر نموذج transformers الافتراضي
