---
### 📄 docs/extras/BLIP/models/blip_nlvr.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج Natural Language Visual Reasoning (NLVR) من BLIP. يُقيّم ما إذا كانت جملة تصف زوجاً من الصور بشكل صحيح. يستخدم سامعين (dual encoders) للصور مع انتباه ذاتي وتبادلي (cross-attention) بين ميزات الصورتين.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BLIP_NLVR(nn.Module)**: نموذج NLVR مع مشفرين للصور ومشفر للنص.
  - `forward(self, image0, image1, caption)`: تشفير الصورتين والنص، حساب الانتباه التبادلي بين الصورتين، إرجاع درجة التصنيف الثنائي.
- **blip_nlvr(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, med_config)**: إنشاء النموذج مع تحميل أوزان pretrained.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/blip.py (BLIP_Base), extras/BLIP/models/med.py, extras/BLIP/models/vit.py
- **يُستدعى بواسطة**: غير مستخدم مباشرة في Fooocus
- **الأثر البرمجي**: غير مؤثر - نموذج احتياطي من BLIP
