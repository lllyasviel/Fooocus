---
### 📄 docs/extras/BLIP/models/blip_vqa.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج Visual Question Answering (VQA) من BLIP. يُولّد إجابات عن أسئلة نصية بناءً على محتوى الصورة. يُحسَّن على مجموعة VQA مع تسلسل إجابات محدود (answer space) أو توليد حر.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BLIP_VQA(nn.Module)**: نموذج الإجابة على الأسئلة المرئية.
  - `forward(self, image, question, answer, alpha, k, weights)`: حساب الخسارة مع ترجيح الإجابات المتعددة.
  - `rank_answer(self, image, question, answer_candidates, ans_len)`: ترتيب إجابات المرشحين حسب درجة التوافق.
- **blip_vqa(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, med_config)**: إنشاء النموذج مع تحميل أوزان pretrained.
- **tile(x, dim, n_tile)**: تكرار tensor عبر بُعد محدد.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/blip.py, extras/BLIP/models/med.py
- **يُستدعى بواسطة**: غير مستخدم مباشرة في Fooocus
- **الأثر البرمجي**: غير مؤثر - نموذج احتياطي من BLIP
