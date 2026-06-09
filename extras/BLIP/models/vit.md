---
### 📄 docs/extras/BLIP/models/vit.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
تنفيذ Vision Transformer (ViT) لمكتبة BLIP. يُقسّم الصورة إلى patches (16×16)، يُمرّرها عبر طبقات Transformer مع انتباه متعدد الرؤوس. يدعم ViT-B/16 (12 طبقة، 768 بُعداً) و ViT-L/14 (24 طبقة، 1024 بُعداً) مع grad checkpointing وخفض معدل التسرب (drop path).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class DropPath(nn.Module)**: تطبيق Drop Path (Stochastic Depth) مع احتمال بقاء عشوائي.
- **class Mlp(nn.Module)**: شبكة MLP (Linear + GELU + Dropout + Linear + Dropout).
- **class Attention(nn.Module)**: انتباه متعدد الرؤوس مع QKV خطي، RoPE، و Dropout.
- **class Block(nn.Module)**: كتلة Transformer (Attention + Mlp + LayerNorm + Residual).
- **class PatchEmbed(nn.Module)**: تقسيم الصورة إلى patches عبر Conv2d (kernel=patch_size, stride=patch_size).
- **class VisionTransformer(nn.Module)**: ViT الكامل: PatchEmbed + CLS token + Position Embedding + Blocks + طبقة الإخراج.
  - `forward(self, x, register_blk)`: تمرير الصورة عبر الشبكة، إرجاع قائمة مخرجات الكتل.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn as nn, functools, einops (rearrange), itertools
- **يُستدعى بواسطة**: extras/BLIP/models/blip.py (create_vit), extras/BLIP/models/blip_nlvr.py
- **الأثر البرمجي**: أساسي لمعالجة الصور في جميع نماذج BLIP
