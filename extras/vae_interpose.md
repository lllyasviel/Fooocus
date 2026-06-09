---
### 📄 docs/extras/vae_interpose.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
تحويل الخوارزميات الكامنة (latent representations) بين فضاءات VAE المختلفة (SDXL ↔ SD 1.x). يستخدم شبكة CNN مع كتل residuals لتكييف الـ latents من SDXL (4×64×64) إلى SD 1.x (4×64×64) مع الحفاظ على البنية والمحتوى.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class ResBlock(nn.Module)**: كتلة متبقية مع BatchNorm، 3 طبقات Conv2d + SiLU، و Dropout.
- **class ExtractBlock(nn.Module)**: كتلة توسيع القنوات: مسار قصير (Conv2d 1×1) + مسار طويل (Conv2d 3×3 + SiLU + Conv2d 3×3).
- **class InterposerModel(nn.Module)**: شبكة interposer الرئيسية: رأس (ExtractBlock)، قلب (Upsample + 6× ResBlocks مع تخطي)، ذيل (Conv2d 4→4).
  - `forward(self, x)`: تمرير الـ latent عبر الرأس، القلب، الذيل، إعادة الخرج.
- **parse(x)**: تحميل النموذج في أول استدعاء (lazy load من safetensors باستخدام model_management)، تحويل الـ latent عبر النموذج مع إدارة تلقائية للـ device/dtype.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: ldm_patched/modules/model_management.py, ldm_patched/modules/model_patcher.py, modules/config.py, safetensors.torch, torch, torch.nn
- **يُستدعى بواسطة**: modules/default_pipeline.py (لربط الـ base model بـ refiner model)
- **الأثر البرمجي**: تعطيله يمنع التبديل بين SDXL و SD 1.x في نموذج التحسين (refiner)
