---
### 📄 docs/extras/BLIP/models/nlvr_encoder.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
مشفر NLVR (Natural Language Visual Reasoning) من BLIP. يُنفّذ انتباهاً تبادلياً بين صورتين (cross-attention بين image0 و image1) قبل تمرير الميزات إلى مشفر النص. يستخدم طبقة Attention محسَّنة مع تفسير (alignment) البعدَيْن الأخيرين.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BertEncoder(nn.Module)**: مشفر مخصص لـ NLVR مع 6 طبقات من الانتباه الذاتي + التبادلي بين الصور.
  - `forward(self, embedding0, embedding1)`: تشفير متوازٍ للصورتين مع cross-attention في كل طبقة.
- **class Embeddings(nn.Module)**: تضمين الصور مع إضافة position embedding.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/med.py (BertLayer, BertAttention, BertIntermediate, BertOutput, BertSelfAttention)، math
- **يُستدعى بواسطة**: extras/BLIP/models/blip_nlvr.py
- **الأثر البرمجي**: غير مؤثر - يستخدم فقط في نموذج BLIP NLVR غير المُستخدم حالياً
