---
### 📄 docs/extras/BLIP/models/blip_pretrain.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج ما قبل التدريب (pretraining) لـ BLIP. يُدرّب على مهام متعددة: فهم الصور (Understanding)، التسمية التوضيحية (Captioning)، والاسترجاع (Retrieval). يستخدم فقدان ITC (Image-Text Contrastive) + ITM (Matching) + LM (Language Modeling).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BLIP_Pretrain(nn.Module)**: النموذج الرئيسي لما قبل التدريب.
  - `forward(self, image, caption, alpha)`: حساب الخسائر المركبة (ITC + ITM + LM).
- **blip_pretrain(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, med_config)**: إنشاء النموذج مع تحميل أوزان pretrained.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/blip.py, extras/BLIP/models/med.py
- **يُستدعى بواسطة**: غير مستخدم مباشرة في Fooocus
- **الأثر البرمجي**: غير مؤثر - نموذج احتياطي من BLIP
