---
### 📄 docs/extras/BLIP/models/blip_itm.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج Image-Text Matching (ITM) من BLIP. يُقيّم التوافق بين الصور والنصوص عبر حساب درجة المطابقة (matching score). يستخدم attention تبادلي بين ميزات الصورة والنص لتحديد أي من النصوص يصف الصورة بشكل أفضل.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BLIP_ITM(nn.Module)**: نموذج ITM مع مشفر BERT ومحول الصور.
  - `forward(self, image, caption, alpha)`: حساب خسارة ITM + خسارة التباين (contrastive loss) مع وزن alpha.
  - `itm_score(self, image, caption)`: حساب درجة المطابقة بين صورة وجملة واحدة.
- **blip_itm(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, med_config)**: إنشاء النموذج مع تحميل أوزان pretrained.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/blip.py (BLIP_Base), extras/BLIP/models/med.py
- **يُستدعى بواسطة**: غير مستخدم مباشرة في Fooocus
- **الأثر البرمجي**: غير مؤثر - نموذج احتياطي من BLIP
