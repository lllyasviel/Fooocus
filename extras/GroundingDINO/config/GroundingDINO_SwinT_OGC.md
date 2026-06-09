---
### 📄 docs/extras/GroundingDINO/config/GroundingDINO_SwinT_OGC.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف تكوين (Config) لنموذج GroundingDINO_SwinT_OGC. يُحدد معمارية النموذج: الـ backbone Swin Transformer (Swin-T)، neck (FPN مع Conv2d)، الـ transformer (6 مشفرات، 6 مفككات)، وإعدادات التدريب. هذا الملف يُستخدم بواسطة load_model من groundingdino لبناء النموذج.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **model_config**: قاموس تكوين يحدد:
  - backbone: Swin-T (window_size=7, embed_dim=96, depths=[2,2,6,2]).
  - neck: ChannelMapper (in_channels [96,192,384,768], out_channels 256, 4 Conv2d 1×1).
  - transformer: 6 encoder layers + 6 decoder layers مع nheads=8، dim_feedforward=2048،d_model=256.
  - matcher: HungarianMatcher (cost_class=2, cost_bbox=5,cost_giou=2).
  - losses: ['labels','boxes','cardinality'].
  - dn_number: 100 (Denoising queries).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: لا يستدعي ملفات (ملف تكوين بحت)
- **يُستدعى بواسطة**: groundingdino.util.slconfig (SLConfig.fromfile) عبر GroundingDINO/util/inference.py
- **الأثر البرمجي**: بدون هذا التكوين لا يمكن بناء نموذج GroundingDINO
